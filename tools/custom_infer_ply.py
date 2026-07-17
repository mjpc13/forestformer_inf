import numpy as np
import torch
from plyfile import PlyData

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData

import oneformer3d


def load_ply_xyz(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]

    points = np.stack(
        [
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ],
        axis=1,
    )

    return points.astype(np.float32)


def build_model(config_path, checkpoint_path, device="cuda"):
    cfg = Config.fromfile(config_path)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def make_batch(points_np, name="sample.ply", device="cuda"):
    points = torch.from_numpy(points_np.astype(np.float32)).to(device)
    n = points.shape[0]

    semantic = np.zeros(n, dtype=np.int64)
    instance = np.zeros(n, dtype=np.int64)

    data_sample = Det3DDataSample()
    data_sample.set_metainfo({
        "lidar_path": name,
    })

    data_sample.eval_ann_info = {
        "pts_semantic_mask": semantic,
        "pts_instance_mask": instance,
    }

    data_sample.gt_pts_seg = PointData(
        pts_semantic_mask=torch.from_numpy(semantic).to(device),
        pts_instance_mask=torch.from_numpy(instance).to(device),
        instance_mask=torch.from_numpy(semantic != 0).to(device),
    )

    batch_inputs_dict = {
        "points": [points],
    }

    batch_data_samples = [data_sample]

    return batch_inputs_dict, batch_data_samples


def load_segmented_ply(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    points = np.stack(
        [
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ],
        axis=1,
    ).astype(np.float32)

    labels = {}
    for name in ("semantic_pred", "instance_pred", "score",
                 "semantic_gt", "instance_gt"):
        if name in names:
            labels[name] = np.asarray(vertex[name])

    return points, labels


def extract_label_array(result):
    pred_pts_seg = result[0].pred_pts_seg
    label_array = pred_pts_seg["label_array"]
    return np.asarray(label_array)


def align_reference_by_xyz(points_np, ref_points, ref_labels, decimals=4):
    pred_keys = np.round(points_np, decimals=decimals)
    ref_keys = np.round(ref_points, decimals=decimals)

    pred_order = np.lexsort((pred_keys[:, 2], pred_keys[:, 1], pred_keys[:, 0]))
    ref_order = np.lexsort((ref_keys[:, 2], ref_keys[:, 1], ref_keys[:, 0]))

    if not np.array_equal(pred_keys[pred_order], ref_keys[ref_order]):
        return ref_points, ref_labels, False

    ref_index_for_pred = np.empty_like(ref_order)
    ref_index_for_pred[pred_order] = ref_order
    aligned_labels = {
        name: values[ref_index_for_pred]
        for name, values in ref_labels.items()
    }
    return ref_points[ref_index_for_pred], aligned_labels, True


def compare_with_segmented_ply(points_np, label_array, reference_path):
    ref_points, ref_labels = load_segmented_ply(reference_path)

    print(f"Comparing predictions with {reference_path}")
    if points_np.shape != ref_points.shape:
        print(f"Point count/shape mismatch: predicted {points_np.shape}, "
              f"reference {ref_points.shape}")
        return

    points_match = np.allclose(points_np, ref_points, atol=1e-4)
    print(f"XYZ coordinates match: {points_match}")
    if not points_match:
        ref_points, ref_labels, aligned = align_reference_by_xyz(
            points_np, ref_points, ref_labels)
        if aligned:
            points_match = np.allclose(points_np, ref_points, atol=1e-4)
            print("Aligned reference rows by XYZ before label comparison.")
            print(f"XYZ coordinates match after alignment: {points_match}")
        else:
            print("Could not align reference rows by XYZ; label comparison "
                  "will stay row-wise.")

    semantic_pred = label_array[:, 0].astype(np.int64)
    instance_pred = label_array[:, 1].astype(np.int64)

    if "semantic_pred" in ref_labels:
        ref_semantic = ref_labels["semantic_pred"].astype(np.int64)
        semantic_mismatches = np.count_nonzero(semantic_pred != ref_semantic)
        print(f"semantic_pred mismatches: {semantic_mismatches}/"
              f"{len(ref_semantic)}")

    if "instance_pred" in ref_labels:
        ref_instance = ref_labels["instance_pred"].astype(np.int64)
        instance_mismatches = np.count_nonzero(instance_pred != ref_instance)
        print(f"instance_pred mismatches: {instance_mismatches}/"
              f"{len(ref_instance)}")

    if label_array.shape[1] > 2 and "score" in ref_labels:
        score_pred = label_array[:, 2].astype(np.float32)
        ref_score = ref_labels["score"].astype(np.float32)
        score_mismatches = np.count_nonzero(
            ~np.isclose(score_pred, ref_score, atol=1e-5))
        print(f"score mismatches: {score_mismatches}/{len(ref_score)}")


def main():
    device = "cuda"

    ply_path = "sample_data/sample.ply"
    reference_ply_path = "sample_data/sample_segmented.ply"
    config_path = "configs/inference_only.py"
    checkpoint_path = "weights/epoch_3000_fix.pth"

    points_np = load_ply_xyz(ply_path)
    print(f"Loaded {points_np.shape[0]} points from {ply_path}")

    model = build_model(config_path, checkpoint_path, device=device)

    batch_inputs_dict, batch_data_samples = make_batch(
        points_np,
        name=ply_path,
        device=device,
    )

    with torch.no_grad():
        result = model.predict(batch_inputs_dict, batch_data_samples)

    print("Inference finished.")
    label_array = extract_label_array(result)
    print(f"Returned label array shape: {label_array.shape}")
    compare_with_segmented_ply(points_np, label_array, reference_ply_path)


if __name__ == "__main__":
    main()
