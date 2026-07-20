# Simplified ForestFormer3D Inference

This repository is a reduced inference-only version of ForestFormer3D. The goal is to load a point cloud directly, build the minimum MMEngine/MMSeg/MinkowskiEngine objects the model still expects, run prediction, and return point-wise labels as NumPy data instead of relying on the original training/data-loading pipeline.

The original project used MMEngine datasets and wrote segmented PLY files inside the model. This fork keeps the model runnable for inference and moves PLY writing into a separate script.

## Repository Layout

```text
configs/inference_only.py       Minimal model config for inference
oneformer3d/                    Model, decoder, sparse UNet, and NMS code
tools/custom_infer_ply.py       Runs inference and compares labels with a reference PLY
tools/write_infer_ply.py        Runs inference and writes sample_data/new_sample_seg.ply
tools/base_modules.py           Small modules required by the model
sample_data/sample.ply          Input sample point cloud
sample_data/sample_segmented.ply Reference segmented point cloud
weights/                        Checkpoint location and local checkpoint file
Dockerfile                      CUDA inference environment
```

## Checkpoint

The inference scripts expect:

```text
weights/epoch_3000_fix.pth
```

The download note is kept in:

```text
weights/checkpointlocation-epoch_3000_fix.pth.txt
```

The checkpoint itself is ignored by Git because it is a large binary artifact.

## Build the Docker Image

Build from the repository root:

```bash
docker build -t simplified-forestformer:debug .
```

The image installs CUDA-enabled PyTorch 1.13.1, MinkowskiEngine, spconv, MMEngine/MMDetection3D dependencies, and CUDA PyG wheels for `torch-scatter` and `torch-cluster`.

## Run Inference

Use `--gpus all` so CUDA is available inside the container:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  simplified-forestformer:debug \
  python tools/custom_infer_ply.py
```

`tools/custom_infer_ply.py` loads `sample_data/sample.ply`, runs the model, extracts the returned label array, and compares it with `sample_data/sample_segmented.ply`.

The returned label array has shape:

```text
N x 3
```

with columns:

```text
semantic_pred, instance_pred, score
```

## Write a Segmented PLY

To write the current inference result to a PLY file:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  simplified-forestformer:debug \
  python tools/write_infer_ply.py
```

By default this writes:

```text
sample_data/new_sample_seg.ply
```

with the same vertex schema as `sample_data/sample_segmented.ply`:

```text
x, y, z, semantic_pred, instance_pred, score, semantic_gt, instance_gt
```

The prediction columns come from the fresh model output. The `semantic_gt` and `instance_gt` columns are copied from the reference PLY when the point count matches; if the reference XYZ order does not match the input XYZ order, the script prints a warning.

You can override paths:

```bash
python tools/write_infer_ply.py \
  --input sample_data/sample.ply \
  --output sample_data/new_sample_seg.ply \
  --reference sample_data/sample_segmented.ply \
  --config configs/inference_only.py \
  --checkpoint weights/epoch_3000_fix.pth \
  --device cuda
```

## Notes

- This code path is inference-only. Training losses, dataloaders, and full MMEngine runner behavior are intentionally not the focus here.
- `sample_data/sample.ply` and `sample_data/sample_segmented.ply` are useful fixtures and are not globally ignored.
- Generated PLY files such as `sample_data/new_sample_seg.ply` are ignored.
- If `torch_cluster.fps` reports `Not compiled with CUDA support`, rebuild the Docker image so the CUDA PyG wheel is installed.
