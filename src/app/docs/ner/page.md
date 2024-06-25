---
title: Named Entity Recognition 
nextjs:
  metadata:
    title: Named Entity Recognition 
    description: Learn about NER.
---

## Overview

{% callout title="Warning" type="warning" %}
Named Entity Recognition is an experimental feature and may be subject to instability. Please be aware that the API and/or functionality could change.
{% /callout %}

Named Entity Recognition is a process of locating and classifying the named entities in a provided text. 


Currently, Scikit-LLM has a single NER estimator (only works with the GPT family) called `Explainable NER`.

Exemplary usage: 

```python
from skllm.models.gpt.tagging.ner import GPTExplainableNER as NER

entities = {
  "PERSON": "A name of an individual.",
  "ORGANIZATION": "A name of a company.",
  "DATE": "A specific time reference."
}

data = [
  "Tim Cook announced new Apple products in San Francisco on June 3, 2022.",
  "Elon Musk visited the Tesla factory in Austin on January 10, 2021.",
  "Mark Zuckerberg introduced Facebook Metaverse in Silicon Valley on May 5, 2023."
]

ner = NER(entities=entities, display_predictions=True)
tagged = ner.fit_transform(data)
```

The model will tag the entities and provide a short reasoning behind its choice. If the `display_predictions` output is set to `True`, the outputs of the model are parsed automatically and presented in a human readable way: each entity is highlighted and the explanation is displayed on hovering over the entity. 

Exemplary output:

==============================
{% html innerHTML="<style>:root{--font-size:16px}.entity{font-size:var(--font-size);padding:2px 4px;border-radius:4px;font-weight:700}@media (prefers-color-scheme:light){.entity-person{background-color:#add8e6;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-person{background-color:#add8e6;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}.entity-organization{background-color:#90ee90;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-organization{background-color:#90ee90;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}.entity-date{background-color:#f08080;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-date{background-color:#f08080;color:#000;border-radius:4px;padding:2px 4px;font-weight:700}}@media (prefers-color-scheme:dark){.entity-person{background-color:#00008b;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-person{background-color:#00008b;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}.entity-organization{background-color:#006400;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-organization{background-color:#006400;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}.entity-date{background-color:#8b0000;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-date{background-color:#8b0000;color:#fff;border-radius:4px;padding:2px 4px;font-weight:700}}</style><div><style>.entity-legend-person-light{background-color:#add8e6;color:#000;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-person-dark{background-color:#00008b;color:#fff;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-person{cursor:pointer;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-organization-light{background-color:#90ee90;color:#000;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-organization-dark{background-color:#006400;color:#fff;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-organization{cursor:pointer;border-radius:4px;padding:2px 4px;font-weight:700}.entity-legend-date-light{background-color:#f08080;color:#000;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-date-dark{background-color:#8b0000;color:#fff;padding:2px 4px;border-radius:4px;font-weight:700}.entity-legend-date{cursor:pointer;border-radius:4px;padding:2px 4px;font-weight:700}</style>Entities:<span class='entity-legend-person' title='PERSON: A name of an individual.' style='margin-right:4px'>PERSON</span><span class='entity-legend-organization' title='ORGANIZATION: A name of a company.' style='margin-right:4px'>ORGANIZATION</span><span class='entity-legend-date' title='DATE: A specific time reference.' style='margin-right:4px'>DATE</span></div><br><span class='entity entity-person' title='PERSON: Tim Cook is the name of an individual, specifically the CEO of Apple.'>Tim Cook</span>announced new<span class='entity entity-organization' title='ORGANIZATION: Apple is the name of a company, specifically a well-known technology company.'>Apple</span>products in San Francisco on<span class='entity entity-date' title='DATE: June 3, 2022 is a specific time reference, indicating a particular date.'>June 3, 2022</span>.<br><span class='entity entity-person' title='PERSON: Elon Musk is a well-known individual, making it a clear example of a PERSON entity.'>Elon Musk</span>visited the<span class='entity entity-organization' title='ORGANIZATION: Tesla is a well-known company, making it a clear example of an ORGANIZATION entity.'>Tesla</span>factory in Austin on<span class='entity entity-date' title='DATE: January 10, 2021 is a specific date, making it a clear example of a DATE entity.'>January 10, 2021</span>.<br><span class='entity entity-person' title='PERSON: Mark Zuckerberg is the name of an individual, specifically the CEO of Facebook.'>Mark Zuckerberg</span>introduced<span class='entity entity-organization' title='ORGANIZATION: Facebook is the name of a company, specifically a social media giant.'>Facebook</span>Metaverse in Silicon Valley on<span class='entity entity-date' title='DATE: May 5, 2023 is a specific time reference, indicating a particular date.'>May 5, 2023</span>.<br>"%}

{% /html %}
==============================

The `display_output` functionality works in both Jupyter Notebook and plain Python scripts. When used outside Jupyter, a new HTML page will be auto-generated and opened in a new browser window.


## Sparse vs Dense NER

We distinguish between two modes of generating the predictions: sparse and dense. 

In dense mode the model produces a complete (tagged) output right away, while in sparse mode only a list of entities is produced which is then mapped to the text via regex.

In most of the scenarios the usage of sparse mode should be preferable for the following reasons: 
  - lower number of output tokens (cheaper to use);
  - strict validation -> it is guaranteed that the output is invertable and only contains the specified entities;
  - higher accuracy, especially with smaller models.

Dense mode should only be used when the following conditions are met:
  - a larger model is used (e.g. gpt-4);
  - the text is expected to contain multiple (distinct) instances of lexically ambiguous words.

For example, in a sentence "**Apple** is the favorite fruit of the CEO of **Apple**", the first and second occurrences of the word "Apple" should be classified as different entities, which is only possible using the dense mode.

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### GPTExplainableNER

```python
from skllm.models.gpt.tagging.ner import GPTExplainableNER
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `entities`      | `dict`  | A dictionary of entities to recognize, with keys as **uppercase** entity names and values as descriptions. |
| `display_predictions`      | `Optional[bool]`  | Determines whether to display predictions, by default False. |
| `sparse_output`      | `Optional[bool]`  | Determines whether to generate a sparse representation of the predictions, by default True. |
| `model`      | `Optional[str]`  | A model to use, by default "gpt-4o". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `num_workers`      | `Optional[int]`  | Number of workers (threads) to use, by default 1. |