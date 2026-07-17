FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        libglib2.0-0 \
        libopenblas-dev \
        libsm6 \
        libxext6 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --no-deps \
        mmengine==0.7.3 \
        mmdet==3.0.0 \
        mmsegmentation==1.0.0 \
        mmcv==2.0.0 \
        spconv-cu116==2.3.6 \
        open3d==0.17.0 \
        scikit-learn==1.2.2 \
        plyfile==1.0.2 \
        tqdm==4.65.0

RUN pip install --no-cache-dir --no-deps --no-index \
        torch-scatter==2.1.0+pt113cu116 \
        torch-cluster==1.6.0+pt113cu116 \
        -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

RUN pip install --no-cache-dir --no-deps \
        git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

RUN apt-get update \
    && apt-get install -y --no-install-recommends libcusolver-dev-11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6" \
    pip install --no-cache-dir --no-deps \
        git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
        --install-option="--blas=openblas" \
        --install-option="--force_cuda"

RUN pip install --no-cache-dir \
        addict==2.4.0 \
        yapf==0.33.0 \
        termcolor==2.3.0 \
        packaging==23.1 \
        rich==13.3.5 \
        opencv-python==4.7.0.72 \
        pycocotools==2.0.6 \
        Shapely==1.8.5 \
        scipy==1.10.1 \
        terminaltables==3.1.10 \
        numba==0.57.0 \
        llvmlite==0.40.0 \
        pccm==0.4.7 \
        ccimport==0.4.2 \
        pybind11==2.10.4 \
        cumm-cu116==0.4.9 \
        pyquaternion==0.9.9 \
        pandas==2.0.1 \
        matplotlib==3.5.2 \
        scikit-image==0.21.0 \
        tensorboard==2.15.1

RUN pip install --no-cache-dir \
    joblib==1.2.0 \
    threadpoolctl==3.1.0

COPY . /workspace

CMD ["python", "tools/custom_infer_ply.py"]
