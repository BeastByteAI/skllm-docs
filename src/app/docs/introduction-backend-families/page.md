---
title: Backend families
nextjs:
  metadata:
    title: Backend families
    description: Overview of backend families.
---
On a high level, Scikit-LLM estimators are divided based on the language model backend family they use. The backend family is defined by the API format and does not necessarily correspond to the language model architecture. For example, all backends that follow the OpenAI API format are groupped into _gpt_ family regardless the actual language model architecture or provider. Eeach backend family has its own set of estimators which are located in the `skllm.models.<family>` sub-module.

For example, the Zero-Shot Classifier is available as `skllm.models.gpt.classification.zero_shot.ZeroShotGPTClassifier` for the _gpt_ family, and as `skllm.models.vertex.classification.zero_shot.ZeroShotVertexClassifier` for the _vertex_ family. The separation between the backend families is necessary to allow for a reasonable level of flexibility if/when model providers introduce model-specific features that are not supported by other providers and hence cannot be easily abstracted away. At the same time, the number of model families is kept to a minimum to simplify the usage and maintenance of the library. Since the OpenAI API is by far the most popular and widely used, backends that follow that format are preferred over the others.

Whenever the backend family supports multiple backends, the default one is used unless the `model` parameter specifies a particular backend namespace. For example, the default backend for the _gpt_ family is the OpenAI backend. However, you can use the Azure backend by setting `model = "azure::<model_name>"`. However, please note that not every estimator supports every backend.

---

## GPT Family

The GPT family includes all backends that follow the OpenAI API format.


### OpenAI (default)

The OpenAI backend is the default backend for the GPT family. It is used whenever the `model` parameter does not specify a particular backend namespace.

To use the OpenAI backend, you need to set your OpenAI API key and organization ID as follows:

```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")
```

### Azure

OpenAI models can be alternatively used as a part of the [Azure OpenAI service](https://azure.microsoft.com/en-us/products/ai-services/openai-service). To use the Azure backend, you need to provide your Azure API key and endpoint as follows:

```python
from skllm.config import SKLLMConfig
# Found under: Resource Management (Left Sidebar) -> Keys and Endpoint -> KEY 1
SKLLMConfig.set_gpt_key("<YOUR_KEY>")
# Found under: Resource Management (Left Sidebar) -> Keys and Endpoint -> Endpoint
SKLLMConfig.set_azure_api_base("<API_BASE>") # e.g. https://<YOUR_PROJECT_NAME>.openai.azure.com/
```

When using the Azure backend, the model should be specified as `model = "azure::<model_deployment_name>"`. For example, if you created a _gpt-3.5_ deployment under the name _my-model_, you should use `model = "azure::my-model"`.

### GGUF

GGUF is an open-source binary file format designed for storing the quantized versions of model weights as well as high level model configurations. GGUF is primarily used in combination with the [Llama CPP](https://github.com/ggerganov/llama.cpp) project, but can also be loaded by some other runtimes.

In order to use GGUF models with scikit-llm, the llama-cpp and its python bindings have to be installed first.

The installation command slightly varies depending on your hardware.

CPU-only, all platforms:
```bash 
pip install scikit-llm[gguf] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir
```

GPU (CUDA 12.1+), Windows/Linux:
```bash 
pip install scikit-llm[gguf] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir
```

GPU (Metal), MacOS, M-series Macs only: 
```bash 
pip install scikit-llm[gguf] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal --no-cache-dir
```

Then, you can use the backend by specifying the model as `model = "gguf::<model_name>"`. The model will be downloaded automatically. 

Currently, the following models are available:

| **Model name** | **Size (GB)**  | Released (Year/Month) | Base model    |
| -------------------- | ---- | ------- |------------------------------------------------------------------------------------------------ |
| gemma2-2b-q6         | 2.3  | 2024/08 |[google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)                              |
| gemma2-9b-q4         | 5.8  | 2024/06 |[google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)                              |
| phi3-mini-q4         | 2.4  | 2024/06 |[microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)      |
| mistral0.3-7b-q4     | 4.4  | 2024/05 |[mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  |
| llama3-8b-q4         | 4.9  | 2024/04 |[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)|

For all of the models, the quantized version is used. The precision is indicated by the suffix in the name (e.g. `q4` stands for 4-bit quantization). By default, we choose the models with 4-bit quantization, but might decide to include models with lower/higher precision as well (for models with higher/lower number of parameters respectively). When picking the model for your use case, the following rule of thumb can be applied: 
- q < 4 : Substantial performance loss, low size
- q = 4 : Optimal trade-off between the loss and the size
- 4 < q < 8 : Minimal performance loss, large size
- q = 8 : Almost no performance loss, very large size

In addition, there exist several quantization schemas of the same precision (e.g. for q4 those can be Q4_0, Q4_K_S, Q4_K_M, etc.). In order to keep it simpler for the users, we omit this information in the model name and select a single sub-type which we consider to be the most optimal.


#### GPU acceleration

GGUF models can be unloaded to a GPU (both fully and partially).

The following command specifies the maximum number of GPU layers: 
```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_gguf_max_gpu_layers(-1)
```
 - 0 : all layers on the CPU
 - -1 : all layers on the GPU
 - n>0 : n layers on the GPU, remaining on the CPU 

Note, that changing the configuration does not reload the model automatically (even if the new estimator is created afterwards). The models can be off-loaded from the memory as follows: 

```python
from skllm.llm.gpt.clients.llama_cpp.handler import ModelCache

ModelCache.clear()
```

This command can also be handy when experimenting with different models in an interactive environment like JupyterNotebook, as the models remain in the memory until the termination of the process.

### Custom URL

Custom URL backend allows to use any GPT estimator with any OpenAI-compatible provider (either running locally or in the cloud).

In order to use the backend, it is necessary to set a global custom url: 
```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_gpt_url("http://localhost:8000/")

clf = ZeroShotGPTClassifier(model="custom_url::<custom_model_name>")
```


{% callout title="Note" %}
When using `custom_url` and `openai` backends within the same script, it is necessary to reset the custom url configuration using `SKLLMConfig.reset_gpt_url()`.
{% /callout %}

---

## Vertex Family

The Vertex family currently includes a single (default) backend, which is the Google Vertex AI.

In order to use the Vertex backend, you need to configure your Google Cloud credentials as follows:

1.  Log in to [Google Cloud Console](https://console.cloud.google.com/) and [create a Google Cloud project](https://developers.google.com/workspace/guides/create-project). After the project is created, select this project from a list of projects next to the Google Cloud logo (upper left corner).
2.  Search for _Vertex AI_ in the search bar and select it from the list of services.
3.  Install a Google Cloud CLI on the local machine by following [the steps from the official documentation](https://cloud.google.com/sdk/docs/install), and [set the application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials#personal) by running the following command:
    ```bash
    gcloud auth application-default login
    ```
4.  Configure Scikit-LLM with your project ID:

    ```python
    from skllm.config import SKLLMConfig

    SKLLMConfig.set_google_project("<YOUR_PROJECT_ID>")
    ```

Additionally, for tuning LLMs in Vertex, it is required to have to have 64 cores of the TPU v3 pod training resource. By default this quota is set to 0 cores and has to be increased as follows (ignore this if you are not planning to use the tunable estimators):

1.  Go to [Quotas](https://cloud.google.com/docs/quota/view-manage#requesting_higher_quota) and filter them for “Restricted image training TPU V3 pod cores per region”.
2.  Select “europe-west4” region (currently this is the only supported region).
3.  Click on “Edit Quotas”, set the limit to 64 and submit the request.
    The request should be approved within a few hours, but it might take up to several days.


## Third party integrations

- [scikit-ollama](https://github.com/AndreasKarasenko/scikit-ollama): Scikit-Ollama provides scikit-llm estimators that allow to use self-hosted LLMs through [Ollama](https://ollama.com/).