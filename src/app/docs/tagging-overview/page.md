---
title: Overview
nextjs:
  metadata:
    title: Overview
    description: Learn about text tagging.
---

Tagging in Scikit-LLM can be an arbitrary task that takes a raw text and returns the same text with inserted XML-like tags. 

For example, a sentiment analysis task could look as follows: 

Input:
```bash
I love my new phone, but I am disappointed with the battery life.
```

Output:
```xml
<positive>I love my new phone,</positive> <negative>but I am disappointed with the battery life.</negative>
```

In an ideal scenario, such tagging process should be invertible, so the original text can always be reconstructed from the tagged one. However, this is not always feasible and hence not considered to be a mandatory requirement.
