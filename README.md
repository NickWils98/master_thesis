# Master Thesis

## Overview

This repository contains the implementation of Input-debias and Neutral-debias for word embeddings. 

### Debiasing Word Embeddings

The `DebiasBert` class provides methods to debias word embeddings. Here is an example of how to use the class:

```python
from debias_class import DebiasBert

# Initialize the debiaser
debiaser = DebiasBert(tokenizer, None, 9, 1)

# Get the original embeddings from your model
old_emb = model.get_input_embeddings()

# Apply full word debiasing
emb = debiaser.debias_full_word_embeddings(old_emb)

# Set the debiased embeddings back to your model
model.set_input_embeddings(emb)
```

Alternatively, to debias only neutral words, provide a list of neutral words when initializing the `DebiasBert` class and use function `debias_word_embeddings`:

```python
from debias_class import DebiasBert

neutral_words = ['example_word1', 'example_word2', ...]
debiaser = DebiasBert(tokenizer, neutral_words, 9, 1)
old_emb = model.get_input_embeddings()
emb = debiaser.debias_word_embeddings(old_emb)
model.set_input_embeddings(emb)
```

### Evaluations and Tests

- **Extrinsic Evaluation**: Files for extrinsic evaluation are located in the `downstream_task` folder. Run these files to evaluate the performance of the debiased embeddings on downstream tasks.
  
- **Intrinsic Evaluation**: The intrinsic evaluation is located in the `run_metrics` file. 

- **Subspace Identification**: The `subspace_identification` file contains code to help determine the ideal number of dimensions for debiasing. Use this to understand the subspace properties of the embeddings.

- **Input-Output Consistency Test**: The `inputOuputConsistencyTest` file contains code to test the consistency of the input embeddings after debiasing. Run this test to ensure the integrity of the embeddings post-debiasing.

## Folder Structure

- `debias_class.py`: Contains the `DebiasBert` class with methods for debiasing word embeddings.
- `downstream_task/`: Contains Files for extrinsic evaluation.
- `run_metrics.py`: Contains intrinsic evaluation.
- `subspace_identification.py`: Contains code to identify the ideal number of dimensions for debiasing.
- `inputOuputConsistencyTest.py`: Contains code to test the consistency of input embeddings after debiasing.
