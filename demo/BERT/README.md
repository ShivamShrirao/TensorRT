# BERT Inference Using TensorRT

This subfolder of the BERT TensorFlow repository, tested and maintained by NVIDIA, provides scripts to perform high-performance inference using NVIDIA TensorRT.


## Table Of Contents

- [Model Overview](#model-overview)
   * [Model Architecture](#model-architecture)
   * [TensorRT Inference Pipeline](#tensorrt-inference-pipeline)
   * [Version Info](#version-info)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
  * [(Optional) Trying a different configuration](#optional-trying-a-different-configuration)
- [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Command-line options](#command-line-options)
  * [TensorRT inference process](#tensorrt-inference-process)
- [Accuracy](#accuracy)
  * [Evaluating Post-Training-Quantization INT8 accuracy](#evaluating-ptq-post-training-quantization-int8-accuracy-using-the-squad-dataset)
  * [Evaluating Quantization-Aware-Training INT8 accuracy](#evaluating-qat-quantization-aware-training-int8-accuracy-using-the-squad-dataset)
- [Experimental](#experimental)
  * [Variable sequence length](#variable-sequence-length)
      * [Run command lines](#run-command-lines)
  * [Sparsity with Quantization Aware Training](#sparsity-with-quantization-aware-training)
      * [Megatron-LM for Question Answering](#megatron-lm-for-question-answering)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
       * [TensorRT inference benchmark](#tensorrt-inference-benchmark)
  * [Results](#results)
    * [Inference performance: NVIDIA A100](#inference-performance-nvidia-a100-40gb)
    * [Inference performance: NVIDIA A30](#inference-performance-nvidia-a30)


## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores for faster inference times while maintaining target accuracy.

Other publicly available implementations of BERT include:
1. [NVIDIA PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)
5. [Google's official implementation](https://github.com/google-research/bert)


### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT:

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feed-forward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERT-Base |12 encoder| 768| 12|4 x  768|512|110M|
|BERT-Large|24 encoder|1024| 16|4 x 1024|512|330M|

Typically, the language model is followed by a few task-specific layers. The model used here includes layers for question answering.

### TensorRT Inference Pipeline

BERT inference consists of three main stages: tokenization, the BERT model, and finally a projection of the tokenized prediction onto the original text.
Since the tokenizer and projection of the final predictions are not nearly as compute-heavy as the model itself, we run them on the host. The BERT model is GPU-accelerated via TensorRT.

The tokenizer splits the input text into tokens that can be consumed by the model. For details on this process, see [this tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_tf_ckpt_large_qa_squad2_amp_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

|Software|Version|
|--------|-------|
|Python|>=3.6|
|TensorRT|8.5.1|
|CUDA|11.6|

## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT OSS build container. If running in a different environment, following packages are required.

* [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
* PyPI Packages:
  * [pycuda](https://pypi.org/project/pycuda/) (tested v2019.1.2)
  * [onnx](https://pypi.org/project/onnx) (tested v1.12.0)
  * [tensorflow](https://pypi.org/project/tensorflow/) (tested v2.9.1)
  * [torch](https://pypi.org/project/torch/) (tested v1.11.0)
* NVIDIA [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU.


## Quick Start Guide

1. Build and launch the container as described in [TensorRT OSS README](https://github.com/NVIDIA/TensorRT/blob/master/README.md).

    **Note:** After this point, all commands should be run from within the container.

2. Verify TensorRT installation by printing the version:
   For example:
    ```bash
    python3 -c "import tensorrt as trt; print(trt.__version__)"
    ```

3. Download the SQuAD dataset and BERT checkpoints:
    ```bash
    cd $TRT_OSSPATH/demo/BERT
    ```

    Download SQuAD v1.1 training and dev dataset.
    ```bash
    bash ./scripts/download_squad.sh
    ```

    Download Tensorflow checkpoints for BERT large model with sequence length 128, fine-tuned for SQuAD v2.0.
    ```bash
    bash scripts/download_model.sh
    ```

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

4. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/model.ckpt -o engines/bert_large_128.engine -b 1 -s 128 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large SQuAD v2 FP16 Sequence Length 128 checkpoint (`-c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1`).

5. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer.
    For example:
    ```bash
    python3 inference.py -e engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
    ```

    b. `inference.ipynb` Jupyter Notebook
    The Jupyter Notebook includes a passage and various example questions and allows you to interactively make modifications and see the outcome.
    To launch the Jupyter Notebook from inside the container, run:
    ```bash
    jupyter notebook --ip 0.0.0.0 inference.ipynb
    ```
    Then, use your browser to open the link displayed. The link should look similar to: `http://127.0.0.1:8888/?token=<TOKEN>`
    
6. Run inference with CUDA Graph support.

    A separate python `inference_c.py` script is provided to run inference with CUDA Graph support. This is necessary since CUDA Graph is only supported through CUDA C/C++ APIs, not pyCUDA. The `inference_c.py` script uses pybind11 to interface with C/C++ for CUDA graph capturing and launching. The cmdline interface is the same as `inference.py` except for an extra `--enable-graph` option.
    
    ```bash
    mkdir -p build; pushd build
    cmake .. -DPYTHON_EXECUTABLE=$(which python)
    make -j
    popd
    python3 inference_c.py -e engines/bert_large_128.engine --enable-graph -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
    ```

    A separate C/C++ inference benchmark executable `perf` (compiled from `perf.cpp`) is provided to run inference benchmarks with CUDA Graph. The cmdline interface is the same as `perf.py` except for an extra `--enable_graph` option.
    
    ```bash
    build/perf -e engines/bert_large_128.engine -b 1 -s 128 -w 100 -i 1000 --enable_graph
    ```
    

### (Optional) Trying a different configuration

If you would like to run another configuration, you can manually download checkpoints using the included script. For example, run:
```bash
bash scripts/download_model.sh base
```
to download a BERT Base model instead of the default BERT Large model.

To view all available model options, run:
```bash
bash scripts/download_model.sh -h
```

## Advanced

The following sections provide greater details on inference with TensorRT.

### Scripts and sample code

In the `root` directory, the most important files are:

- `builder.py` - Builds an engine for the specified BERT model
- `Dockerfile` - Container which includes dependencies and model checkpoints to run BERT
- `inference.ipynb` - Runs inference interactively
- `inference.py` - Runs inference with a given passage and question
- `perf.py` - Runs inference benchmarks

The `scripts/` folder encapsulates all the one-click scripts required for running various supported functionalities, such as:

- `build.sh` - Builds a Docker container that is ready to run BERT
- `launch.sh` - Launches the container created by the `build.sh` script.
- `download_model.sh` - Downloads pre-trained model checkpoints from NGC
- `inference_benchmark.sh` - Runs an inference benchmark and prints results

Other folders included in the `root` directory are:

- `helpers` - Contains helpers for tokenization of inputs

The `infer_c/` folder contains all the necessary C/C++ files required for CUDA Graph support.
- `bert_infer.h` - Defines necessary data structures for running BERT inference
- `infer_c.cpp` - Defines C/C++ interface using pybind11 that can be plugged into `inference_c.py`
- `perf.cpp` - Runs inference benchmarks. It is equivalent to `perf.py`, with an extra option `--enable_graph` to enable CUDA Graph support.

### Command-line options

To view the available parameters for each script, you can use the help flag (`-h`).

### TensorRT inference process

As mentioned in the [Quick Start Guide](#quick-start-guide), two options are provided for running inference:
1. The `inference.py` script which accepts a passage and a question and then runs the engine to generate an answer. Alternatively, this script can be used to run inference on the Squad dataset.
2. The `inference.ipynb` Jupyter Notebook which includes a passage and various example questions and allows you to interactively make modifications and see the outcome.

## Accuracy

### Evaluating PTQ (post-training quantization) Int8 Accuracy Using The SQuAD Dataset
1.  Download Tensorflow checkpoints for a BERT Large FP16 SQuAD v2 model with a sequence length of 384:
    ```bash
    bash scripts/download_model.sh large 384 v2
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/train-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```
### Evaluating QAT (quantization aware training) Int8 Accuracy Using The SQuAD Dataset
1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:
    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx 
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:

    ```bash
    python3 inference.py -e engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```
## Experimental

### Variable sequence length
In our prior implementation, we used inputs padded to max length along with corresponding input masks to handle variable sequence length inputs in a batch. The padding results in some wasted computations which can be avoided by handling variable sequence length inputs natively. Now we have a new approach called the variable sequence length method. By concatenating each input id into a single long input id, and concatenating each input segment id into a single long segment id, TensorRT can know the exact starts and ends by providing an extra sequence length buffer that contains the start and end positions of each sequence. Now we can eliminate the wasted computation in the input paddings.

Note this is an experimental feature because we only support Xavier+ GPUs, also there is neither FP32 support nor INT8 PTQ calibration.

1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:
    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2. Build an engine:

    **FP16 engine**
    ```bash
    mkdir -p engines && python3 builder_varseqlen.py -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o engines/bert_varseq_fp16.engine -b 1 -s 64 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 64 (`-s 64`) using FP16 precision computation where possible (`--fp16`).


    **INT8 engine**
    ```bash
    mkdir -p engines && python3 builder_varseqlen.py -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o engines/bert_varseq_int8.engine -b 1 -s 256 --int8 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 256 (`-s 256`) using INT8 precision computation where possible (`--int8`).

3. Run inference 

    Evaluate the F1 score and exact match score using the squad dataset:
    
    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -s 256 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

    Run the quesion and answer mode:

    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
    ```

4. Collect performance data

    ```bash
    python3 perf_varseqlen.py -e engines/bert_varseq_int8.engine -b 1 -s 256
    ```

    This will collect performance data run use batch size 1 (`-b 1`) and sequence length of 256 (`-s 256`). 

5. Collect performance data with CUDA graph enabled

    We can use the same `inference_c.py` and `build/perf` to collect performance data with cuda graph enabled. The command line is the same as run without variable sequence length. 

### Sparsity with Quantization Aware Training

Fine-grained 2:4 structured sparsity support introduced in NVIDIA Ampere GPUs can produce significant performance gains in BERT inference. The network is first trained using dense weights, then fine-grained structured pruning is applied, and finally the remaining non-zero weights are fine-tuned with additional training steps. This method results in virtually no loss in inferencing accuracy.

Using INT8 precision with quantization scales obtained from Post-Training Quantization (PTQ) can produce additional performance gains, but may also result in accuracy loss. Alternatively, for PyTorch-trained models, NVIDIA [PyTorch-Quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) can be leveraged to perform quantized fine tuning (a.k.a. Quantization Aware Training or QAT) and generate the INT8 quantization scales as part of training. This generally results in higher accuracy compared to PTQ.

To demonstrate the potential speedups from these optimizations in demoBERT, we provide the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) transformer model finetuned for SQuAD 2.0 task with sparsity and quantization.

The sparse weights are generated by finetuning with INT8 Quantization Aware Training recipe. This feature can be used with the fixed or variable sequence length implementations by passing in `-sp` flag to demoBERT builder.

#### Megatron-LM for Question Answering

##### Example: Megatron-LM Large SQuAD v2.0 with sparse weights for sequence length 384

**Build the TensorRT engine**:

Options specified:
* `--megatron`  : assume Megatron style residuals instead of vanilla BERT.
* `--pickle`    : specify a pickle file containing the PyTorch statedict corresponding to fine-tuned Megatron model.
* `-sp`         : enable sparsity during engine optimization and treat the weights as sparse.
* `--int8 --il` : enable int8 tactics/plugins with interleaving.

```bash
bash ./scripts/download_model.sh 384 v1_1 # BERT-large model checkpoint fine-tuned for SQuAD 1.1
bash ./scripts/download_model.sh pyt megatron-large int8-qat sparse # Megatron-LM model weights
export CKPT_PATH=models/fine-tuned/bert_pyt_statedict_megatron_sparse_int8qat_v21.03.0/bert_pyt_statedict_megatron_sparse_int8_qat
mkdir -p engines && python3 builder_varseqlen.py -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -b 1 -s 384 -o engines/megatron_large_seqlen384_int8qat_sparse.engine --fp16 --int8 --strict -il --megatron --pickle $CKPT_PATH -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -sp
```

**Ask a question**:
```bash
python3 inference_varseqlen.py -e engines/megatron_large_seqlen384_int8qat_sparse.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
```

**Evaluate F1 score**:
```bash
python3 inference_varseqlen.py -e engines/megatron_large_seqlen384_int8qat_sparse.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
```
Expected output:
```
&&&& PASSED TensorRT BERT Squad Accuracy matches reference.
{"exact_match": 84.03973509933775, "f1": 90.88667129897755}
```

## Performance

### Benchmarking

The following section shows how to run the inference benchmarks for BERT.

#### TensorRT inference benchmark

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which takes the following steps for each set of model parameters:

1. Downloads checkpoints and builds a TensorRT engine if it does not already exist.

2. Runs 100 warm-up iteration then runs inference for 1000 to 2000 iterations for each batch size specified in the script, selecting the profile best for each size.

**Note:** The time measurements do not include the time required to copy inputs to the device and copy outputs to the host.

To run the inference benchmark script, run:
```bash
bash scripts/inference_benchmark.sh --gpu <arch>
```
Options for `<arch>` are: 'Volta', 'Xavier', 'Turing', 'Ampere'

Note: Some of the configurations in the benchmark script require 16GB of GPU memory. On GPUs with smaller amounts of memory, parts of the benchmark may fail to run.

Also note that BERT Large engines, especially using mixed precision with large batch sizes and sequence lengths may take a couple hours to build.

### Results

The following sections provide details on how we achieved our performance and inference.

#### Inference performance: NVIDIA A100 (40GB)

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A100 (40G).

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.69 | 0.70 | 0.57 | 0.78 | 0.78 | 0.63 |
| 128 | 2 | 0.77 | 0.78 | 0.62 | 0.88 | 0.88 | 0.74 |
| 128 | 4 | 0.74 | 0.94 | 0.74 | 0.93 | 0.93 | 0.93 |
| 128 | 8 | 0.95 | 1.22 | 0.96 | 1.32 | 1.32 | 1.32 |
| 128 | 12 | 1.28 | 1.28 | 1.21 | 1.71 | 1.72 | 1.70 |
| 128 | 16 | 1.34 | 1.34 | 1.34 | 2.08 | 2.08 | 2.07 |
| 128 | 24 | 1.84 | 1.84 | 1.83 | 3.07 | 3.07 | 3.02 |
| 128 | 32 | 2.26 | 2.26 | 2.25 | 3.94 | 3.98 | 3.90 |
| 128 | 64 | 4.14 | 4.21 | 4.14 | 7.65 | 7.66 | 7.59 |
| 128 | 128 | 8.13 | 8.13 | 8.05 | 15.34 | 15.43 | 15.21 |
| 384 | 1 | 1.28 | 1.28 | 1.14 | 1.39 | 1.39 | 1.30 |
| 384 | 2 | 1.32 | 1.33 | 1.32 | 1.55 | 1.55 | 1.55 |
| 384 | 4 | 1.67 | 1.67 | 1.67 | 2.10 | 2.11 | 2.10 |
| 384 | 8 | 2.22 | 2.22 | 2.22 | 3.36 | 3.36 | 3.33 |
| 384 | 12 | 3.34 | 3.34 | 3.34 | 4.80 | 4.84 | 4.79 |
| 384 | 16 | 4.03 | 4.03 | 4.03 | 6.39 | 6.40 | 6.36 |
| 384 | 24 | 5.74 | 5.74 | 5.73 | 9.36 | 9.43 | 9.31 |
| 384 | 32 | 7.68 | 7.68 | 7.67 | 12.90 | 12.91 | 12.84 |
| 384 | 64 | 14.88 | 14.93 | 14.77 | 25.03 | 25.22 | 24.83 |
| 384 | 128 | 29.05 | 29.07 | 28.78 | 49.02 | 49.14 | 48.65 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.33 | 1.33 | 1.25 | 1.58 | 1.70 | 1.58 |
| 128 | 2 | 1.44 | 1.44 | 1.43 | 1.97 | 1.98 | 1.97 |
| 128 | 4 | 1.78 | 2.19 | 1.79 | 2.55 | 2.56 | 2.54 |
| 128 | 8 | 2.66 | 2.66 | 2.66 | 3.93 | 3.93 | 3.92 |
| 128 | 12 | 3.11 | 3.11 | 3.10 | 5.07 | 5.08 | 5.02 |
| 128 | 16 | 4.06 | 4.06 | 4.05 | 6.94 | 6.95 | 6.86 |
| 128 | 24 | 5.31 | 5.35 | 5.30 | 9.69 | 9.70 | 9.59 |
| 128 | 32 | 7.05 | 7.05 | 6.99 | 12.96 | 13.01 | 12.87 |
| 128 | 64 | 12.93 | 12.94 | 12.79 | 24.85 | 24.86 | 24.61 |
| 128 | 128 | 25.15 | 25.16 | 25.09 | 49.22 | 49.39 | 48.76 |
| 384 | 1 | 2.56 | 2.57 | 2.56 | 2.98 | 2.99 | 2.98 |
| 384 | 2 | 3.06 | 3.06 | 3.06 | 3.92 | 3.92 | 3.92 |
| 384 | 4 | 4.03 | 4.04 | 4.03 | 5.75 | 5.75 | 5.73 |
| 384 | 8 | 7.20 | 7.21 | 7.20 | 11.04 | 11.05 | 10.96 |
| 384 | 12 | 9.19 | 9.19 | 9.18 | 15.48 | 15.49 | 15.34 |
| 384 | 16 | 12.34 | 12.35 | 12.33 | 21.22 | 21.22 | 20.97 |
| 384 | 24 | 17.75 | 17.75 | 17.60 | 31.09 | 31.12 | 30.75 |
| 384 | 32 | 23.27 | 23.30 | 23.04 | 40.92 | 41.25 | 40.64 |
| 384 | 64 | 44.97 | 44.98 | 44.88 | 79.43 | 79.66 | 79.21 |
| 384 | 128 | 88.18 | 88.33 | 87.92 | 156.77 | 157.18 | 155.75 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.15 | 1.47 | 1.16 |
| 128 | 2 | 1.38 | 1.64 | 1.39 |
| 128 | 4 | 1.79 | 1.86 | 1.79 |
| 128 | 8 | 2.54 | 2.54 | 2.53 |
| 128 | 12 | 2.95 | 2.95 | 2.95 |
| 128 | 16 | 3.98 | 3.98 | 3.97 |
| 128 | 24 | 4.92 | 4.92 | 4.91 |
| 128 | 32 | 6.85 | 6.87 | 6.81 |
| 128 | 64 | 11.63 | 11.63 | 11.61 |
| 128 | 128 | 21.27 | 21.29 | 21.04 |
| 384 | 1 | 1.72 | 1.78 | 1.72 |
| 384 | 2 | 2.21 | 2.21 | 2.21 |
| 384 | 4 | 3.48 | 3.48 | 3.47 |
| 384 | 8 | 5.76 | 5.76 | 5.76 |
| 384 | 12 | 8.38 | 8.39 | 8.37 |
| 384 | 16 | 10.38 | 10.39 | 10.37 |
| 384 | 24 | 14.59 | 14.60 | 14.58 |
| 384 | 32 | 18.77 | 18.78 | 18.76 |
| 384 | 64 | 35.82 | 35.84 | 35.50 |
| 384 | 128 | 67.65 | 67.81 | 67.42 |

#### Inference performance: NVIDIA A30

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A30.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.88 | 0.88 | 0.61 | 0.80 | 0.80 | 0.79 |
| 128 | 2 | 0.75 | 1.13 | 0.76 | 1.08 | 1.08 | 0.98 |
| 128 | 4 | 1.04 | 1.56 | 1.05 | 1.43 | 1.64 | 1.41 |
| 128 | 8 | 1.44 | 1.47 | 1.42 | 2.40 | 2.44 | 2.38 |
| 128 | 12 | 1.92 | 1.93 | 1.91 | 3.40 | 3.41 | 3.37 |
| 128 | 16 | 2.38 | 2.45 | 2.35 | 4.33 | 4.34 | 4.28 |
| 128 | 24 | 3.47 | 3.54 | 3.42 | 6.54 | 6.57 | 6.45 |
| 128 | 32 | 4.37 | 4.46 | 4.36 | 8.34 | 8.42 | 8.30 |
| 128 | 64 | 8.57 | 8.60 | 8.45 | 16.43 | 16.53 | 16.29 |
| 128 | 128 | 16.44 | 16.62 | 16.34 | 31.99 | 32.31 | 31.91 |
| 384 | 1 | 1.31 | 1.31 | 1.31 | 1.63 | 1.64 | 1.63 |
| 384 | 2 | 1.66 | 1.66 | 1.66 | 2.26 | 2.29 | 2.25 |
| 384 | 4 | 2.29 | 2.29 | 2.26 | 3.76 | 3.78 | 3.69 |
| 384 | 8 | 4.23 | 4.26 | 4.18 | 7.18 | 7.22 | 7.16 |
| 384 | 12 | 6.05 | 6.11 | 5.97 | 10.20 | 10.27 | 10.09 |
| 384 | 16 | 8.06 | 8.08 | 7.99 | 13.90 | 13.92 | 13.73 |
| 384 | 24 | 11.78 | 11.85 | 11.65 | 20.24 | 20.35 | 20.07 |
| 384 | 32 | 15.37 | 15.37 | 15.21 | 26.79 | 26.95 | 26.65 |
| 384 | 64 | 30.49 | 30.69 | 30.16 | 52.04 | 52.31 | 51.66 |
| 384 | 128 | 59.95 | 60.29 | 59.42 | 102.76 | 103.17 | 102.24 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.46 | 1.64 | 1.46 | 2.00 | 2.00 | 2.00 |
| 128 | 2 | 1.84 | 1.84 | 1.83 | 2.77 | 2.78 | 2.74 |
| 128 | 4 | 2.68 | 2.69 | 2.68 | 4.36 | 4.40 | 4.32 |
| 128 | 8 | 4.28 | 4.30 | 4.26 | 8.08 | 8.13 | 7.98 |
| 128 | 12 | 5.60 | 5.78 | 5.59 | 10.58 | 10.66 | 10.47 |
| 128 | 16 | 7.59 | 7.66 | 7.51 | 14.67 | 14.76 | 14.58 |
| 128 | 24 | 10.52 | 10.60 | 10.40 | 20.55 | 20.64 | 20.34 |
| 128 | 32 | 14.05 | 14.14 | 13.91 | 28.14 | 28.26 | 27.89 |
| 128 | 64 | 26.76 | 26.89 | 26.47 | 53.44 | 53.73 | 53.13 |
| 128 | 128 | 52.23 | 52.45 | 51.67 | 105.72 | 106.15 | 105.14 |
| 384 | 1 | 3.33 | 3.33 | 3.33 | 4.21 | 4.27 | 4.19 |
| 384 | 2 | 4.22 | 4.25 | 4.20 | 6.58 | 6.60 | 6.54 |
| 384 | 4 | 7.26 | 7.28 | 7.21 | 11.88 | 11.95 | 11.85 |
| 384 | 8 | 12.91 | 12.92 | 12.77 | 22.74 | 22.76 | 22.50 |
| 384 | 12 | 18.60 | 18.73 | 18.42 | 33.43 | 33.64 | 33.15 |
| 384 | 16 | 24.06 | 24.23 | 23.93 | 44.35 | 44.51 | 43.90 |
| 384 | 24 | 35.57 | 35.81 | 35.29 | 65.13 | 65.60 | 64.81 |
| 384 | 32 | 47.01 | 47.13 | 46.53 | 85.44 | 85.94 | 84.79 |
| 384 | 64 | 91.69 | 92.11 | 91.08 | 168.65 | 168.92 | 167.77 |
| 384 | 128 | 180.15 | 180.50 | 179.30 | 330.86 | 331.38 | 329.67 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.45 | 1.55 | 1.45 |
| 128 | 2 | 1.87 | 2.10 | 1.87 |
| 128 | 4 | 2.76 | 2.77 | 2.75 |
| 128 | 8 | 4.14 | 4.14 | 4.13 |
| 128 | 12 | 5.22 | 5.22 | 5.21 |
| 128 | 16 | 7.42 | 7.43 | 7.41 |
| 128 | 24 | 9.97 | 9.99 | 9.86 |
| 128 | 32 | 12.84 | 12.88 | 12.76 |
| 128 | 64 | 24.33 | 24.45 | 24.08 |
| 128 | 128 | 46.19 | 46.37 | 45.68 |
| 384 | 1 | 2.37 | 2.37 | 2.37 |
| 384 | 2 | 3.88 | 3.89 | 3.87 |
| 384 | 4 | 6.05 | 6.06 | 6.05 |
| 384 | 8 | 11.52 | 11.54 | 11.45 |
| 384 | 12 | 15.73 | 15.77 | 15.60 |
| 384 | 16 | 20.95 | 21.01 | 20.85 |
| 384 | 24 | 29.97 | 30.01 | 29.71 |
| 384 | 32 | 39.94 | 40.03 | 39.63 |
| 384 | 64 | 76.57 | 76.81 | 76.20 |
| 384 | 128 | 148.66 | 149.18 | 147.85 |

