---
title: Quick start
---

Scikit-LLM allows you to seamlessly integrate powerful language models into scikit-learn for enhanced text analysis tasks. 

Let's see how it is possible to use Scikit-LLM to perform zero-shot text classification with GPT-4.

---

## Installation

First of all, it is necessary to install Scikit-LLM. You can do it by running the following command:

```bash
pip install scikit-llm
```

---

## API Key Configuration

For this example, we will use GPT-4, which requires an OpenAI API key. You can get one [here](https://platform.openai.com/api-keys).

Once you have your API key, you can set it as follows:

```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")
```

{% callout title="Note" %}
Scikit-LLM supports other language models, including the locally hosted ones. For more information, please refer to the [Backend Families](/docs/introduction-backend-families) section.
{% /callout %}

---

## Zero-Shot Text Classification

Now, we are ready to perform zero-shot text classification with GPT-4. Let's start by loading a sample dataset:

```python
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()
```

Then, we can create a classifier instance and fit it using conventional scikit-learn syntax:

```python
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

clf = ZeroShotGPTClassifier(model="gpt-4-turbo")
clf.fit(X,y)
clf.predict(X)
```

Scikit-LLM will automatically query the OpenAI API and transform the response into a regular list of labels.

Additionally, Scikit-LLM will ensure that the obtained response contains a valid label. If this is not the case, a label will be selected randomly (label probabilities are proportional to label occurrences in the "training" set).

Furthermore, since the "training" data is not strictly required, it can be fully omitted. The only thing that has to be provided is the list of candidate labels.

```python
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(None, ["positive", "negative", "neutral"])
clf.predict(X)
```