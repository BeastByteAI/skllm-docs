---
title: Few-shot text classification
nextjs:
  metadata:
    title: Few-shot text classification
    description: Learn about few-shot text classification.
---

## Overview

Few-shot text classification is a task of classifying a text into one of the pre-defined classes based on a few examples of each class. For example, given a few examples of the class _positive_, _negative_, and _neutral_, the model should be able to classify a new text into one of these classes.

The estimators provided by Scikit-LLM do not automatically select the subset of the training data, and instead use the entire training set to construct the examples. Therefore, if your training set is large, you might want to consider splitting it into training and validation sets, while keeping the training set small (we recommend not to exceed 10 examples per class).

Example using GPT-4:

```python
from skllm.models.gpt.classification.few_shot import (
FewShotGPTClassifier,
MultiLabelFewShotGPTClassifier,
)
from skllm.datasets import (
    get_classification_dataset,
    get_multilabel_classification_dataset,
)

# single label
X, y = get_classification_dataset()
clf = FewShotGPTClassifier(model="gpt-4")
clf.fit(X,y)
labels = clf.predict(X)

# multi-label
X, y = get_multilabel_classification_dataset()
clf = MultiLabelFewShotGPTClassifier(max_labels=2, model="gpt-4")
clf.fit(X,y)
labels = clf.predict(X)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn classifier.

### FewShotGPTClassifier
```python
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |

### MultiLabelFewShotGPTClassifier
```python
from skllm.models.gpt.classification.few_shot import MultiLabelFewShotGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `max_labels`      | `Optional[int]`  | Maximum labels per sample, by default 5. |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |