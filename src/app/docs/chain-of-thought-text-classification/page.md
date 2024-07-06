---
title: Chain-of-thought text classification
nextjs:
  metadata:
    title: Chain-of-thought text classification
    description: Learn about chain-of-thought text classification.
---

## Overview

Chain-of-thought text classification is similar to zero-shot classification since it does not require any labeled data beforehand. The only difference is that, in addition to the label itself, the model generates some additional reasoning behind its choice. In some cases, such an approach might lead to much better performance, but at the cost of higher token consumption.

Example using GPT-4o:

```python
from skllm.models.gpt.classification.zero_shot import CoTGPTClassifier
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()

clf = CoTGPTClassifier(model="gpt-4o")
clf.fit(X,y)
predictions = clf.predict(X)
labels, reasoning = predictions[:, 0], predictions[:, 1]
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn classifier.

### CoTGPTClassifier
```python
from skllm.models.gpt.classification.zero_shot import CoTGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
