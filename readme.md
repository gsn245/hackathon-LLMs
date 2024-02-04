# Starting kit - LLMs

Some loose ideas: Write a bot that has an interface to arxiv/google scholar and
1. reviews papers (maybe using supervised information from OpenReview).
2. answers the question "has this idea been done already?"
3. given a methods section, writes the related work section.

## Setting up your environment

Create a conda environment and activate it

```bash
conda create -n llms python=3.10
conda activate llms
```

Install the requirements by running
```bash
pip install -r requirements.txt
```

Afterwards, you can install this package with
```bash
pip install -e .
```

## Basic examples

### A simple query to GPT-3.5/4

```python

```

### Downloading and reading a pdf

```python

from llms import read_pdf

... = read_pdf(...)

```

### Passing a pdf to GPT-3.5/4 and asking a question

### Downloading pdfs from a conference in OpenReview

### Querying papers from arxiv

### Computing embeddings of different papers

## Intermediate examples

### Chatting with a pdf

### Reviewing a paper w. an LLM

```python
# TODO: write

...
```

### TODOs

- Write an interface with OpenReview and arXiv
- Build a simple prompt-driven review.

## Advanced

### Having an LLM use your Python functions

- 🧑‍🍳 Notebook from the OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb


## Example: writing a related work section
