FROM nvcr.io/nvidia/tensorrt:22.10-py3

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade tensorrt

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /workspace

ENV CUDA_MODULE_LOADING="LAZY"

ENV PLUGIN_LIBS="/workspace/build/out/libnvinfer_plugin.so"