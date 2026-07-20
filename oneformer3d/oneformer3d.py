import torch
import spconv.pytorch as spconv
import MinkowskiEngine as ME

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms
import numpy as np
from tools.base_modules import Seq, MLP
from torch_cluster import fps
from tqdm import tqdm


@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery(Base3DDetector):
    r"""FOR-instance dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 min_spatial_shape,
                 query_point_num=200,
                 backbone=None,
                 decoder=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 radius=16,
                 score_th=0.4,
                 chunk=20_000,
                 fast_voxel_mapping=True,
                 return_panoptic=True,
                 region_step_divisor=4):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        self.Embed = Seq().append(MLP([num_channels, num_channels], bias=False))
        self.Embed.append(torch.nn.Linear(num_channels, 5))
        self.query_point_num = query_point_num
        self.radius = radius
        self.score_th = score_th
        self.chunk = chunk
        self.fast_voxel_mapping = fast_voxel_mapping
        self.return_panoptic = return_panoptic
        self.region_step_divisor = region_step_divisor
        self.BiSemantic = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 2))
            .append(torch.nn.LogSoftmax(dim=-1))
        )

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    @staticmethod
    def make_label_array(semantic_pred, instance_pred, scores=None):
        """Return point-wise labels as an Nx2/Nx3 NumPy array."""
        semantic_pred = np.asarray(semantic_pred, dtype=np.int64)
        instance_pred = np.asarray(instance_pred, dtype=np.int64)
        if scores is None:
            return np.column_stack((semantic_pred, instance_pred))
        scores = np.asarray(scores, dtype=np.float32)
        return np.column_stack((semantic_pred, instance_pred, scores))

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        raise NotImplementedError(
            "This packaged model is inference-only; training loss code is not available.")


    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        step_size = self.radius / self.region_step_divisor
        grid_size = 0.2
        num_points = 640000
        radius_sq = self.radius ** 2
        edge_radius_sq = (self.radius - 0.5) ** 2
        original_points = batch_inputs_dict['points'][0]
        regions = self.generate_cylindrical_regions(
            original_points, self.radius, step_size)
        point_ids = torch.arange(
            original_points.shape[0],
            device=original_points.device,
            dtype=torch.long)
        votes_counter = torch.zeros(
            (original_points.shape[0], self.test_cfg.num_sem_cls),
            dtype=torch.int32,
            device=original_points.device)
        votes_counter_flat = votes_counter.view(-1)
        all_pre_ins = torch.full(
            (original_points.shape[0],),
            -1,
            dtype=torch.long,
            device=original_points.device)
        global_instance_scores = torch.zeros(
            (original_points.shape[0],),
            dtype=torch.float32,
            device=original_points.device)

        max_instance = 0
        best_mask_chunks = []
        score_th1 = self.score_th
        score_th2 = 0.3
        last_results = None
        last_originids = None

        for region in tqdm(regions, desc="Processing regions"):
            region_mask = (
                (original_points[:, 0] - region[0]) ** 2
                + (original_points[:, 1] - region[1]) ** 2
            ) <= radius_sq
            pc1 = original_points[region_mask]
            pc1_indices = point_ids[region_mask]

            if len(pc1) == 0:
                continue

            pc2, pc2_indices, pc2_inverse = self.grid_sample(
                pc1, pc1_indices, grid_size, return_inverse=True)
            if len(pc2) <= num_points:
                pc3 = pc2
                pc3_indices = pc2_indices
                if self.fast_voxel_mapping:
                    nn_idx_pc1 = pc2_inverse
                else:
                    nn_idx_pc1 = None
            else:
                pc3, pc3_indices = self.points_random_sampling(
                    pc2, pc2_indices, num_points)
                nn_idx_pc1 = None

            coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))
            x = self.extract_feat(x)

            embed_logits = self.Embed(x[0])
            bi_semantic_logits = self.BiSemantic(x[0])

            semantic_predictions_bi = torch.argmax(bi_semantic_logits, dim=1)
            tree_indices = torch.nonzero(
                semantic_predictions_bi == 1, as_tuple=False).flatten()

            if nn_idx_pc1 is None:
                nn_idx_pc1 = self.nearest_sample_indices(pc1, pc3)

            if tree_indices.numel() > 1:
                batch_tensor = torch.zeros(
                    embed_logits[tree_indices].size(0),
                    dtype=torch.long,
                    device=embed_logits.device)
                ratio = min(
                    self.query_point_num / embed_logits[tree_indices].size(0),
                    1.0)
                selected_indices_case4 = tree_indices[
                    fps(embed_logits[tree_indices], batch_tensor, ratio=ratio)]

                x = self.decoder(x, [x[0][selected_indices_case4]])
                results_list = self.predict_by_feat_test(
                    x, inverse_mapping2, pc3, selected_indices_case4,
                    return_tensors=True,
                    include_panoptic=self.return_panoptic)

                masks = results_list[0].pts_instance_mask[0].to(pc3.device).bool()
                scores = results_list[0].instance_scores.to(pc3.device)

                if masks.size(0) > 0:
                    pc3_distances_sq = (
                        (pc3[:, 0] - region[0]) ** 2
                        + (pc3[:, 1] - region[1]) ** 2)
                    edge_points = pc3_distances_sq > edge_radius_sq
                    touches_edge = (masks & edge_points.unsqueeze(0)).any(dim=1)
                    valid_mask = (scores > score_th1) & ~touches_edge
                    masks_kept = masks[valid_mask]
                    scores_kept = scores[valid_mask]

                    if masks_kept.size(0) > 0:
                        rows_list = []
                        cols_list = []
                        max_chunk = (
                            torch.iinfo(torch.int32).max // masks_kept.shape[0]
                        ) - 1_000_000
                        chunk_size = min(nn_idx_pc1.shape[0], max(1, max_chunk))

                        for start in range(0, nn_idx_pc1.shape[0], chunk_size):
                            end = min(start + chunk_size, nn_idx_pc1.shape[0])
                            rows_chunk, cols_chunk = masks_kept[:, nn_idx_pc1[start:end]].nonzero(
                                as_tuple=True)
                            rows_list.append(rows_chunk)
                            cols_list.append(cols_chunk + start)

                        rows = torch.cat(rows_list, dim=0)
                        cols = torch.cat(cols_list, dim=0)
                        if rows.numel() > 0:
                            score_per_hit = scores_kept[rows]
                            best_score = torch.full(
                                (pc1.shape[0],), -1., device=pc3.device)
                            best_mid = torch.full(
                                (pc1.shape[0],),
                                -1,
                                dtype=torch.long,
                                device=pc3.device)

                            best_score.index_reduce_(0, cols, score_per_hit, reduce='amax')
                            improved_mask = score_per_hit == best_score[cols]
                            best_mid.index_put_(
                                (cols[improved_mask],),
                                rows[improved_mask],
                                accumulate=False)

                            better_pts = best_score > global_instance_scores[pc1_indices]
                            better_ids = pc1_indices[better_pts]
                            global_instance_scores[better_ids] = best_score[better_pts]
                            all_pre_ins[better_ids] = max_instance + best_mid[better_pts]
                            best_mask_chunks.append((
                                pc1_indices[cols].detach(),
                                rows.detach(),
                                scores_kept.detach(),
                                max_instance))

                        max_instance += int(masks_kept.size(0))

                sem_pred_pc3 = results_list[0].pts_semantic_mask[0]
                cylinder_current_semantic_pre = sem_pred_pc3.to(pc3.device)[
                    nn_idx_pc1].long()
                self.add_semantic_votes(
                    votes_counter_flat, pc1_indices,
                    cylinder_current_semantic_pre, self.test_cfg.num_sem_cls)
                last_results = results_list
                last_originids = pc3_indices.detach()
            else:
                sem_pred_pc3 = torch.argmax(
                    bi_semantic_logits[inverse_mapping2], dim=1)
                cylinder_current_semantic_pre = sem_pred_pc3.to(pc3.device)[
                    nn_idx_pc1].long()
                self.add_semantic_votes(
                    votes_counter_flat, pc1_indices,
                    cylinder_current_semantic_pre, self.test_cfg.num_sem_cls)

        final_semantic_labels_t = votes_counter.argmax(1)
        final_semantic_labels_t[votes_counter.sum(1) == 0] = -1
        all_pre_ins[final_semantic_labels_t == 0] = -1
        final_semantic_labels = final_semantic_labels_t.cpu().numpy()
        all_pre_ins = all_pre_ins.cpu().numpy()

        best_masks = []
        for mask_points_t, mask_rows_t, mask_scores_t, instance_base in best_mask_chunks:
            if mask_points_t.numel() == 0:
                continue

            mask_points = mask_points_t.cpu().numpy()
            mask_rows = mask_rows_t.cpu().numpy()
            mask_scores = mask_scores_t.cpu().numpy()

            for mid in np.unique(mask_rows):
                best_masks.append((
                    mask_points[mask_rows == mid],
                    instance_base + int(mid),
                    float(mask_scores[int(mid)])))

        if torch.is_tensor(last_originids):
            last_originids = last_originids.cpu().numpy()

        if last_results is not None:
            def _to_numpy(value):
                if torch.is_tensor(value):
                    return value.detach().cpu().numpy()
                return value

            last_results = [
                PointData(
                    pts_semantic_mask=[
                        _to_numpy(mask)
                        for mask in result.pts_semantic_mask],
                    pts_instance_mask=[
                        _to_numpy(mask)
                        for mask in result.pts_instance_mask],
                    instance_labels=_to_numpy(result.instance_labels),
                    instance_scores=_to_numpy(result.instance_scores),
                    query_select_voxel_idx=_to_numpy(result.query_select_voxel_idx),
                    query_select_voxel_idx2=_to_numpy(result.query_select_voxel_idx2))
                for result in last_results]

        uniq, cnt = np.unique(all_pre_ins, return_counts=True)
        to_kill = np.isin(all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])
        all_pre_ins[to_kill] = -1

        unique_best_masks = []
        for mask_points, instance_id, score in best_masks:
            if np.any(all_pre_ins[mask_points] == instance_id):
                unique_best_masks.append((mask_points, instance_id, score))

        clean_all_pre_ins, _, merged_instance_scores = (
            self.merge_overlapping_instances_by_score_speedup(
                all_pre_ins, unique_best_masks, overlap_threshold=score_th2))
        unique_labels = np.unique(clean_all_pre_ins)
        unique_labels = unique_labels[unique_labels >= 0]
        relabel_map = {
            old_label: new_label
            for new_label, old_label in enumerate(unique_labels)}
        relabel_map[-1] = -1
        clean_all_pre_ins = np.vectorize(relabel_map.get)(clean_all_pre_ins)
        label_array = self.make_label_array(
            final_semantic_labels,
            clean_all_pre_ins,
            merged_instance_scores)

        if last_results is not None and len(last_results) == len(batch_data_samples):
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = last_results[i]
                data_sample.pred_pts_seg['originids'] = last_originids
                data_sample.pred_pts_seg['semantic_pred'] = final_semantic_labels
                data_sample.pred_pts_seg['instance_pred'] = clean_all_pre_ins
                data_sample.pred_pts_seg['score'] = merged_instance_scores
                data_sample.pred_pts_seg['label_array'] = label_array
        else:
            for data_sample in batch_data_samples:
                data_sample.pred_pts_seg = PointData(
                    semantic_pred=final_semantic_labels,
                    instance_pred=clean_all_pre_ins,
                    score=merged_instance_scores,
                    label_array=label_array)

        return batch_data_samples



    def predict_by_feat_test(self, out, superpoints, coordinates, queries,
                             return_tensors=False, include_panoptic=True):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)

        ground_points = coordinates[sem_res == 0]
        if ground_points.size(0) > 0:
            ground_z_max = ground_points[:, 2].max()
        else:
            ground_z_max = coordinates.new_tensor(float('inf'))

        inst_res = self.pred_inst_sem_test(
            pred_masks[:-self.test_cfg.num_sem_cls, :],
            pred_scores[:-self.test_cfg.num_sem_cls, :],
            superpoints, self.test_cfg.inst_score_thr, sem_res, coordinates,
            ground_z_max, queries)
        if include_panoptic:
            inst_res_for_pan = None
            if self.test_cfg.inst_score_thr == self.test_cfg.pan_score_thr:
                inst_res_for_pan = inst_res
            pan_res = self.pred_pan_sem(
                pred_masks, pred_scores, superpoints, sem_res, coordinates,
                ground_z_max, queries, inst_res_for_pan)
        else:
            pan_res = None

        if return_tensors:
            pts_semantic_mask = [sem_res]
            pts_instance_mask = [inst_res[0].bool()]
            instance_labels = inst_res[1]
            instance_scores = inst_res[2]
            query_select_voxel_idx = inst_res[3]
            query_select_voxel_idx2 = queries.new_empty((0,), dtype=torch.long)
            if pan_res is not None:
                pts_semantic_mask.append(pan_res[0])
                pts_instance_mask.append(pan_res[1])
                query_select_voxel_idx2 = pan_res[2]
        else:
            pts_semantic_mask = [sem_res.cpu().numpy()]
            pts_instance_mask = [inst_res[0].cpu().bool().numpy()]
            instance_labels = inst_res[1].cpu().numpy()
            instance_scores = inst_res[2].cpu().numpy()
            query_select_voxel_idx = inst_res[3].cpu().numpy()
            query_select_voxel_idx2 = np.empty((0,), dtype=np.int64)
            if pan_res is not None:
                pts_semantic_mask.append(pan_res[0].cpu().numpy())
                pts_instance_mask.append(pan_res[1].cpu().numpy())
                query_select_voxel_idx2 = pan_res[2].cpu().numpy()

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                query_select_voxel_idx=query_select_voxel_idx,
                query_select_voxel_idx2=query_select_voxel_idx2)]

    def pred_inst_sem_test(self, pred_masks, pred_scores,
                           superpoints, score_threshold, sem_res, coordinates,
                           ground_z_max, queries):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scores of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                queries.shape[0],
                1).flatten(0, 1)

        scores, topk_idx = scores.flatten(0, 1).topk(
            min(self.test_cfg.topk_insts, queries.shape[0]), sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        queries_select = queries[topk_idx]
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        queries_select = queries_select[keep_inds]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        num_points_in_mask = mask_pred.sum(dim=1)
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        if mask_pred.size(0) > 0:
            z_values = coordinates[:, 2].unsqueeze(0).expand(mask_pred.size(0), -1)
            min_z = z_values.masked_fill(~mask_pred, float('inf')).min(dim=1).values
            has_points = mask_pred.any(dim=1)
            scores = scores.masked_fill((~has_points) | (min_z > ground_z_max + 5), 0)

        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries_select = queries_select[score_mask]

        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries_select = queries_select[npoint_mask]

        return mask_pred, labels, scores, queries_select

    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan_sem(self, pred_masks, pred_scores,
                     superpoints, sem_res, coordinates, ground_z_max, queries,
                     inst_res=None):
        """Predict panoptic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        if inst_res is None:
            mask_pred, labels, scores, queries_select = self.pred_inst_sem_test(
                pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
                superpoints, thr, sem_res, coordinates, ground_z_max, queries)
        else:
            mask_pred, labels, scores, queries_select = inst_res

        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)

        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]
        queries_select = queries_select[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map, queries_select

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]
        queries_select = queries_select[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_counts = torch.bincount(
            things_inst_mask.long(),
            minlength=mask_pred.shape[0] + 1)
        remove_inst = inst_counts <= self.test_cfg.npoint_thr
        remove_inst[0] = False
        things_inst_mask = things_inst_mask.masked_fill(
            remove_inst[things_inst_mask.long()], 0)
        queries_retained = ~remove_inst[1:]

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0

        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask

        queries_select = queries_select[queries_retained]
        return sem_map_src_mapping, sem_map, queries_select

    @staticmethod
    def generate_cylindrical_regions(points, radius, step_size):
        x_min, x_max = points[:, 0].amin().item(), points[:, 0].amax().item()
        y_min, y_max = points[:, 1].amin().item(), points[:, 1].amax().item()

        regions = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                regions.append((x, y))
                y += step_size
            x += step_size

        return regions

    @staticmethod
    def grid_sample(points: torch.Tensor,
                    indices: torch.Tensor,
                    grid_size: float,
                    return_inverse: bool = False):
        """
        Voxel-downsample point cloud by averaging points in each voxel.

        Args
        ----
        points  : (N, 3)  xyz or xyzf  GPU tensor
        indices : (N,)    original indices (int64 / int32) GPU tensor
        grid_size : float voxel size

        Returns
        -------
        vox_points  : (M, 3)   averaged coords per voxel (same dtype/device)
        vox_indices : (M,)     one representative original index per voxel
        """
        voxel = torch.floor(points / grid_size).to(torch.int32)

        uniq, inverse = torch.unique(voxel, return_inverse=True, dim=0)
        M = uniq.size(0)

        ones = torch.ones_like(inverse, dtype=points.dtype)

        sum_xyz = torch.zeros((M, points.size(1)), device=points.device, dtype=points.dtype)
        cnt_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)

        sum_xyz.index_add_(0, inverse, points)
        cnt_xyz.index_add_(0, inverse, ones)

        vox_points = sum_xyz / cnt_xyz.unsqueeze(1)

        vox_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
        vox_indices.index_copy_(0, inverse, indices)

        if return_inverse:
            return vox_points, vox_indices, inverse
        return vox_points, vox_indices

    def nearest_sample_indices(self, points, samples):
        nn_idx = []
        samples = samples.float()
        with torch.no_grad():
            for start in range(0, points.shape[0], self.chunk):
                end = min(start + self.chunk, points.shape[0])
                nn_idx.append(
                    torch.cdist(points[start:end].float(), samples).argmin(1))
        return torch.cat(nn_idx)

    @staticmethod
    def add_semantic_votes(votes_counter_flat, point_indices, semantic_labels,
                           num_semantic_classes):
        flat_indices = point_indices * num_semantic_classes + semantic_labels
        votes_counter_flat.index_add_(
            0, flat_indices,
            torch.ones_like(semantic_labels, dtype=votes_counter_flat.dtype))

    @staticmethod
    def points_random_sampling(points, indices, num_points):
        choices = torch.randperm(
            points.shape[0], device=points.device)[:num_points]
        sampled_points = points[choices]
        sampled_indices = indices[choices]
        return sampled_points, sampled_indices


    @staticmethod
    def merge_overlapping_instances_by_score_speedup(all_pre_ins,
                                            best_masks,
                                            overlap_threshold=0.30):

        N = all_pre_ins.shape[0]

        merged_instance_labels  = np.full(N, -1, dtype=int)
        merged_instance_scores  = np.full(N, -1.0, dtype=float)

        if not best_masks:
            return merged_instance_labels, [], merged_instance_scores

        best_masks = sorted(best_masks, key=lambda x: -x[2])

        taken_flag = np.zeros(N, dtype=np.bool_)
        kept_masks = []

        for pts_idx, inst_id, score in best_masks:
            pts_idx = np.asarray(pts_idx, dtype=int)
            overlap = taken_flag[pts_idx].mean()

            if overlap > overlap_threshold:
                continue

            kept_masks.append((pts_idx, inst_id, score))

            merged_instance_labels[pts_idx] = inst_id
            merged_instance_scores[pts_idx] = score
            taken_flag[pts_idx]             = True

        return merged_instance_labels, kept_masks, merged_instance_scores
