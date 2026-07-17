import numpy as np
import torch
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures import PointData


def numpy_sample_to_model_inputs(sample, device="cuda"):
    points_np = sample["points"].astype(np.float32)

    # The model expects a list of point tensors.
    points = torch.from_numpy(points_np).to(device)

    n = points_np.shape[0]

    semantic = sample.get(
        "pts_semantic_mask",
        np.zeros(n, dtype=np.int64)
    )
    instance = sample.get(
        "pts_instance_mask",
        np.zeros(n, dtype=np.int64)
    )

    data_sample = Det3DDataSample()
    data_sample.set_metainfo({
        "lidar_path": sample.get("name", "numpy_input.bin"),
    })

    # The current predict() reads these fields.
    data_sample.eval_ann_info = {
        "pts_semantic_mask": semantic,
        "pts_instance_mask": instance,
    }

    # Not always used in inference, but useful for compatibility.
    data_sample.gt_pts_seg = PointData(
        pts_semantic_mask=torch.from_numpy(semantic).to(device),
        pts_instance_mask=torch.from_numpy(instance).to(device),
        instance_mask=torch.from_numpy(semantic != 0).to(device),
    )

    batch_inputs_dict = {
        "points": [points]
    }

    batch_data_samples = [data_sample]

    return batch_inputs_dict, batch_data_samples