python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt


export TRT_OSSPATH=/workspace

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"
export CUDA_MODULE_LOADING="LAZY"


cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt