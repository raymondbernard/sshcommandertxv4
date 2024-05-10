# 🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙

<p align="center">
<img src="https://gitlab-master.nvidia.com/winai/trt-llm-rag-windows/-/raw/main/media/rag-demo.gif"  align="center">
</p>

ChatRTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes. Leveraging retrieval-augmented generation (RAG), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), and RTX acceleration, you can query a custom chatbot to quickly get contextually relevant answers. This app also lets you give query through your voice and lets you retreive images matching your voice or text input. And because it all runs locally on your Windows RTX PC or workstation, you’ll get fast and secure results.
Chat with RTX supports various file formats, including text, pdf, doc/docx, xml, png, jpg, bmp. Simply point the application at the folder containing your files and it'll load them into the library in a matter of seconds.

The AI models that are supported in this app:
- LLaMa 2 13B
- Mistral 7B
- ChatGLM3 6B
- Whisper Medium (for supporting voice input)
- CLIP (for images)

The pipeline incorporates the above AI models, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [LlamaIndex](https://www.llamaindex.ai/) and the [FAISS](https://github.com/facebookresearch/faiss) vector search library. For demonstration, the dataset consists of recent articles sourced from [NVIDIA Gefore News](https://www.nvidia.com/en-us/geforce/news/).


### What is RAG? 🔍
Retrieval-augmented generation (RAG) for large language models (LLMs) seeks to enhance prediction accuracy by leveraging an external datastore during inference. This approach constructs a comprehensive prompt enriched with context, historical data, and recent or relevant knowledge.

## Getting Started

### Hardware requirement
- Chat with RTX is currently built for RTX 3xxx and RTX 4xxx series GPUs that have at least 8GB of GPU memory.
- 50 GB of available hard disk space
- Windows 10/11
- Latest NVIDIA GPU drivers

### Installer

If you are using ChatRTX installer, setup of the models selected during installation is done by the installer. You can skip the Setup Steps below, start the installed 'NVIDIA ChatRTX' from desktop icon, and refer to [Use additional model](#use-additional-model) for additional models.

### Setup Steps

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) 0.9v for Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows). Ensure you have the pre-requisites in place.

Command:
```
pip install tensorrt_llm==0.9 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```
Prerequisites 
- [Python 3.10](https://www.python.org/downloads/windows/)
- [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Windows&target_arch=x86_64)
- [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
- [cuDNN](https://developer.nvidia.com/cudnn)

More details in trt-llm page

2. Clone this repo into a local dir (%ChatRTX Folder%), and install necessary libraries
```
git clone https://github.com/NVIDIA/trt-llm-rag-windows.git
cd trt-llm-rag-windows
pip install -r requirements.txt
```

3. In this project, the AWQ int4 quantized models of LLM are used. Before using it, you'll need to build a TensorRT Engine specific to your GPU. Below are the step to build the engine

- **Get tokenizer, assets:** 

From [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) repository on Huggingface, download config.json, tokenizer.json, tokenizer.model, tokenizer_config.json. Place all these tokenizer files in dir "%ChatRTX Folder%\model\mistral_model\tokenizer"

Download [mel_filters.npz](https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz), [multilingual.toktoken](https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken) and place into dir "%ChatRTX Folder%\model\whisper\whisper_assets"

- **Get weights:** 

Download the config.json and rank0.safetensor (Mistral 7B int4 quantized model weights) from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/llama/models/mistral-7b-int4-chat/files?version=1.2). Place the files in dir %ChatRTX Folder%\model\mistral_model\model_checkpoints

Download the Whisper medium weights [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt). Place it into "%ChatRTX Folder%\model\whisper\whisper_assets".

- **Get TRT-LLM exmaple repo**:

Download [TRT-LLM 0.9v](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.9.0) repo to build the engine. This directory will be be referred to as %TensorRT-LLM folder%.

- **Build TRT engine:** 

Below are the commands to build the engines. Navigate into %TensorRT-LLM folder% and run following commands.

Mistral 7B int4:
```
trtllm-build --checkpoint_dir %checkpoints_local_dir% --output_dir %ChatRTX Folder%\model\mistral_model\engine --gpt_attention_plugin float16 --gemm_plugin float16 --max_batch_size 1 --max_input_len 7168 --max_output_len 1024 --context_fmha=enable --paged_kv_cache=disable --remove_input_padding=disable
```

Whisper int8:
```
python %TensorRT-LLM folder%\examples\whisper\build.py --output_dir %ChatRTX Folder%\model\whisper\whisper_medium_int8_engine --use_gpt_attention_plugin --use_gemm_plugin --use_bert_attention_plugin --enable_context_fmha --max_batch_size 1 --max_beam_width 1 --model_name medium --use_weight_only --model_dir %ChatRTX Folder%\model\whisper\whisper_assets
```

Building above two models are sufficient to run the app. Other models can be downloaded and built after running the app following instructions in ['Use additional model'](#use-additional-model).

## Using App

- ### Run App
```
python verify_install.py
python app.py
```


- ### Use additional model
1. In the app UI that gets launched in browser after running app.py, click on 'Add new models' in the 'AI model' section.
2. Select the model from drop down list, read the model license and check the box of 'License'
3. Click on 'Download models' icon to start the download of model files in the background.
4. After downloading finishes, click on the newly appearing button 'Install'. This will build the TRT LLM engine files if necessary.
5. The installed model will now show up in the 'Select AI model' drop down list.

- ### Deleting model
In case any model is not needed, model can be removed by:
1. Clicking on the gear icon on the top right of the UI.
2. Clicking on 'Delete AI model' icon adjacent to the model name.

## Using your own data
- By default this app loads data from the dataset/ directory into the vector store. To use your own data select the folder in the 'Dataset' section of UI.


This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
