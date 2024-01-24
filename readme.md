# Starting kit - LLMs

Some loose ideas: Write a bot that has an interface to arxiv/google scholar and
- reviews papers (maybe using supervised information from OpenReview).
- answers the question "has this idea been done already?"
- given a methods section, writes the related work section.

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

## Getting an LLM to use your functions

- 🧑‍🍳 Notebook from the OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb

## Example: Having an LLM review a paper

## Example: Knowledge retrieval from arXiv

- 🧑‍🍳 Notebook from the OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb


## Example: writing a related work section
