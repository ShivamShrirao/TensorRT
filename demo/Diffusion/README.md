# Introduction

```bash
git submodule update --init --recursive
cd demo/Diffusion
docker build -t tensorrt .
docker run -it --rm --gpus=all -v $PWD/../../:/workspace tensorrt /bin/bash
. ./demo/Diffusion/build_plugin.sh
```
### HuggingFace user access token

To download the model checkpoints for the Stable Diffusion pipeline, you will need a `read` access token. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image guided by a single text prompt

```bash
LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" -v --model_name_or_path="src/weights-66000" --build-preview-features --hf-token=$HF_TOKEN
```

```bash
LD_PRELOAD=${PLUGIN_LIBS} python3 trt_infer_pipeline.py "a woman wearing orange shirt" -v --model_name_or_path="src/weights-66000"
```
