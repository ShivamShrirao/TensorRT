cd /workspace
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

cd /workspace/demo/Diffusion
mkdir -p onnx engine output