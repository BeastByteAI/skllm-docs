---
title: Text translation
nextjs:
  metadata:
    title: Text translation
    description: Learn about text translation.
---

## Overview

LLMs have proven their proficiency in translation tasks. To leverage this capability, Scikit-LLM provides the Translator module, designed for translating any given text into a specified target language.

Example:

```python
from skllm.models.gpt.text2text.translation import GPTTranslator
from skllm.datasets import get_translation_dataset

X = get_translation_dataset()
t = GPTTranslator(model="gpt-3.5-turbo", output_language="English")
translated_text = t.fit_transform(X)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### GPTTranslator
```python
from skllm.models.gpt.text2text.translation import GPTTranslator
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `output_language`      | `str`  | Target language, by default "English". |