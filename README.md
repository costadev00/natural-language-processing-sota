# natural-language-processing-sota

A collection of Jupyter notebook implementations for modern Natural Language Processing, inspired by the conceptual progression of Stanford CS224N. This repository combines practical implementations with the original papers that shaped the evolution of representation learning in NLP.

## Overview

This repository was created to organize important NLP implementations together with the papers that introduced or refined the underlying ideas. The goal is not only to store code, but also to document the historical and conceptual progression of the field.

The project begins with foundational ideas about language representation, moves through the main embedding methods introduced in the Word2Vec and GloVe era, discusses evaluation challenges, and then advances toward the Transformer architecture.

## Repository Structure

```text
natural-language-processing-sota/
├── README.md
├── implementations/
│   ├── cbow.ipynb
│   ├── skip_gram.ipynb
│   ├── skip_gram_negative_sampling.ipynb
│   ├── glove.ipynb
│   └── transformer.ipynb
└── papers/
    ├── 01_human_language_understanding_and_reasoning.pdf
    ├── 02_efficient_estimation_of_word_representation_in_vector_space.pdf
    ├── 03_distributed_representations_of_words_and_phrases_and_their_compositionality.pdf
    ├── 04_glove_global_vectors_for_word_representation.pdf
    └── 05_evaluation_methods_for_unsupervised_word_embeddings.pdf
```
## Papers

### 1. Human Language Understanding and Reasoning
This paper provides the conceptual foundation for the repository. It discusses how human knowledge and reasoning can be represented computationally through language, and why language understanding is a central problem in artificial intelligence. It helps frame the broader motivation behind Natural Language Processing before moving into specific embedding models.

### 2. Efficient Estimation of Word Representations in Vector Space
This paper is one of the first major neural works on learning word embeddings at scale. It introduces the two main Word2Vec architectures, Continuous Bag of Words (CBOW) and Skip Gram, showing how words can be represented as dense vectors that capture semantic and syntactic relationships efficiently.

**Related implementations:**
* `implementations/cbow.ipynb`
* `implementations/skip_gram.ipynb`

### 3. Distributed Representations of Words and Phrases and their Compositionality
This paper improves the original Skip Gram model with techniques such as negative sampling and other practical optimizations that make training more efficient and scalable. It also extends representation learning beyond single words by incorporating phrases and analyzing how embeddings capture analogical and compositional relationships.

**Related implementation:**
* `implementations/skip_gram_negative_sampling.ipynb`

### 4. GloVe: Global Vectors for Word Representation
GloVe proposes an alternative method for learning word embeddings based on global word co-occurrence statistics. Instead of relying only on local context prediction, it uses a weighted objective over a co-occurrence matrix, aiming to capture meaningful semantic structure through a more global view of language.

**Related implementation:**
* `implementations/glove.ipynb`

### 5. Evaluation Methods for Unsupervised Word Embeddings
This paper critiques the idea that embedding quality can be adequately measured using only similarity scores or geometric distances in vector space. It argues that evaluating embeddings is a more complex problem, since different tasks, datasets, and benchmarks may favor different representations. In this sense, embedding evaluation is itself an important research topic.
```
---

## Implementations
This repository currently contains the following notebook implementations:

* **CBOW**: Implementation of the Continuous Bag of Words architecture for learning word embeddings from surrounding context words.
* **Skip Gram**: Implementation of the Skip Gram architecture for predicting surrounding words from a target word.
* **Skip Gram with Negative Sampling**: Improved Skip Gram implementation using negative sampling and training optimizations for better efficiency.
* **GloVe**: Implementation of the GloVe training pipeline based on global co-occurrence statistics.
* **Transformer**: Implementation of the Transformer architecture, representing the transition from classical word embeddings to modern sequence modeling and the foundations of large language models.

---

## Conceptual Progression
The repository follows a conceptual progression through the development of NLP methods:

1. Language as a medium for representing knowledge
2. Word embeddings through predictive neural models
3. Efficient training improvements for embedding learning
4. Global statistical methods for vector representations
5. Critical perspectives on embedding evaluation
6. Transition to Transformer-based architectures

---

## Purpose of This Repository
This repository is intended to serve as:

* A study and implementation archive for important NLP algorithms
* A structured reference for the historical evolution of representation learning
* A practical complement to the papers included in the repository
* A technical portfolio of notebook-based NLP implementations
    