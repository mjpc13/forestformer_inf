import argparse
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from custom_infer_ply import (
    build_model,
    extract_label_array,
    load_ply_xyz,
    make_batch,
)


OUTPUT_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("semantic_pred", "i4"),
    ("instance_pred", "i4"),
    ("score", "f4"),
    ("semantic_gt", "i4"),
    ("instance_gt", "i4"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and write predictions to a segmented PLY.")
    parser.add_argument("--input", default="sample_data/sample.ply")
    parser.add_argument("--output", default="sample_data/new_sample_seg.ply")
    parser.add_argument("--reference", default="sample_data/sample_segmented.ply")
    parser.add_argument("--config", default="configs/inference_only.py")
    parser.add_argument("--checkpoint", default="weights/epoch_3000_fix.pth")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_reference_gt(reference_path, points_np):
    n_points = points_np.shape[0]
    semantic_gt = np.zeros(n_points, dtype=np.int32)
    instance_gt = np.zeros(n_points, dtype=np.int32)
    text_output = True

    if not reference_path:
        print("No reference PLY provided; semantic_gt/instance_gt set to zero.")
        return semantic_gt, instance_gt, text_output

    if not os.path.exists(reference_path):
        print(f"Reference PLY not found: {reference_path}")
        print("semantic_gt/instance_gt set to zero.")
        return semantic_gt, instance_gt, text_output

    reference_ply = PlyData.read(reference_path)
    reference_vertex = reference_ply["vertex"]
    reference_names = reference_vertex.data.dtype.names
    text_output = getattr(reference_ply, "text", True)

    if len(reference_vertex) != n_points:
        print(
            f"Reference point count mismatch: {len(reference_vertex)} vs "
            f"{n_points}. semantic_gt/instance_gt set to zero.")
        return semantic_gt, instance_gt, text_output

    if all(name in reference_names for name in ("x", "y", "z")):
        reference_points = np.stack(
            [
                np.asarray(reference_vertex["x"]),
                np.asarray(reference_vertex["y"]),
                np.asarray(reference_vertex["z"]),
            ],
            axis=1,
        ).astype(np.float32)
        if not np.allclose(points_np, reference_points, atol=1e-4):
            print(
                "Warning: reference XYZ does not match input XYZ row-wise; "
                "copying GT columns by row only.")

    if "semantic_gt" in reference_names:
        semantic_gt = np.asarray(reference_vertex["semantic_gt"],
                                 dtype=np.int32)
    else:
        print("Reference PLY has no semantic_gt column; using zeros.")

    if "instance_gt" in reference_names:
        instance_gt = np.asarray(reference_vertex["instance_gt"],
                                 dtype=np.int32)
    else:
        print("Reference PLY has no instance_gt column; using zeros.")

    return semantic_gt, instance_gt, text_output


def write_segmented_ply(points_np, label_array, output_path, semantic_gt,
                        instance_gt, text=True):
    label_array = np.asarray(label_array)
    if label_array.shape[0] != points_np.shape[0]:
        raise ValueError(
            f"Point/label count mismatch: {points_np.shape[0]} points vs "
            f"{label_array.shape[0]} labels")

    semantic_pred = label_array[:, 0].astype(np.int32)
    instance_pred = label_array[:, 1].astype(np.int32)
    if label_array.shape[1] > 2:
        scores = label_array[:, 2].astype(np.float32)
    else:
        scores = np.full(points_np.shape[0], -1.0, dtype=np.float32)

    vertex = np.empty(points_np.shape[0], dtype=OUTPUT_DTYPE)
    vertex["x"] = points_np[:, 0].astype(np.float64)
    vertex["y"] = points_np[:, 1].astype(np.float64)
    vertex["z"] = points_np[:, 2].astype(np.float64)
    vertex["semantic_pred"] = semantic_pred
    vertex["instance_pred"] = instance_pred
    vertex["score"] = scores
    vertex["semantic_gt"] = semantic_gt.astype(np.int32)
    vertex["instance_gt"] = instance_gt.astype(np.int32)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(
        output_path)


def main():
    args = parse_args()

    points_np = load_ply_xyz(args.input)
    print(f"Loaded {points_np.shape[0]} points from {args.input}")

    model = build_model(args.config, args.checkpoint, device=args.device)
    batch_inputs_dict, batch_data_samples = make_batch(
        points_np,
        name=args.input,
        device=args.device,
    )

    with torch.no_grad():
        result = model.predict(batch_inputs_dict, batch_data_samples)

    label_array = extract_label_array(result)
    semantic_gt, instance_gt, text_output = load_reference_gt(
        args.reference, points_np)
    write_segmented_ply(
        points_np,
        label_array,
        args.output,
        semantic_gt,
        instance_gt,
        text=text_output,
    )

    print(f"Wrote segmented PLY to {args.output}")
    print("Columns: " + ", ".join(name for name, _ in OUTPUT_DTYPE))


if __name__ == "__main__":
    main()
