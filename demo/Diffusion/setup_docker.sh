python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt

cd /workspace
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="/workspace/build/out/libnvinfer_plugin.so"
export CUDA_MODULE_LOADING="LAZY"

cd /workspace/demo/Diffusion
pip3 install -r requirements.txt