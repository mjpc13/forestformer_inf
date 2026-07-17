import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData

import oneformer3d


def build_model(config_path, checkpoint_path, device='cuda'):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def make_batch(sample, device='cuda'):
    points_np = sample['points'].astype(np.float32)
    points = torch.from_numpy(points_np).to(device)

    n = len(points_np)
    semantic = sample.get('pts_semantic_mask', np.zeros(n, dtype=np.int64))
    instance = sample.get('pts_instance_mask', np.zeros(n, dtype=np.int64))

    data_sample = Det3DDataSample()
    data_sample.set_metainfo({
        'lidar_path': sample.get('name', 'numpy_input.bin'),
    })

    data_sample.eval_ann_info = {
        'pts_semantic_mask': semantic,
        'pts_instance_mask': instance,
    }

    data_sample.gt_pts_seg = PointData(
        pts_semantic_mask=torch.from_numpy(semantic).to(device),
        pts_instance_mask=torch.from_numpy(instance).to(device),
        instance_mask=torch.from_numpy(semantic != 0).to(device),
    )

    return {'points': [points]}, [data_sample]


def infer(model, sample, device='cuda'):
    batch_inputs_dict, batch_data_samples = make_batch(sample, device)

    with torch.no_grad():
        return model.predict(batch_inputs_dict, batch_data_samples)


if __name__ == '__main__':
    model = build_model(
        'configs/inference_only.py',
        'weights/epoch_3000_fix.pth',
    )

    sample = {
        'name': 'scan_001',
        'points': np.load('points.npy'),  # [N, 3]
    }

    result = infer(model, sample)