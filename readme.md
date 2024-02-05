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
from openai import OpenAI

from llms.api import load_api_key, compute_price

API_KEY = load_api_key()  # Modify it by adding your key here, or get the-key from Miguel.
GPT_MODEL_NAME = "gpt-3.5-turbo-0125"

client = OpenAI(api_key=API_KEY)

completion = client.chat.completions.create(
    model=GPT_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are BayesGPT, a helpful assistant who is an expert on probabilistic modeling and Machine Learning.",
        },
        {
            "role": "user",
            "content": "What is an intuitive description of the Kullback-Leibler divergence?",
        },
    ],
)

print(completion.choices[0].message.content)
print(
    f"input tokens: {completion.usage.prompt_tokens}, "
    f"output tokens: {completion.usage.completion_tokens})"
    "\n"
    f"Cost: {compute_price(completion)} USD"
)
```

### Reading a pdf

```python
from pathlib import Path

from llms.utils.pdf import read_pdf

ROOT_DIR = ...

# Load the pdf as a string
paper = read_pdf(ROOT_DIR / "data" / "raw" / "example_pdfs" / "score_based.pdf")

print(score_based[:1000])
```

### Passing a pdf to GPT-3.5/4 and asking for a summary

```python
from pathlib import Path

from openai import OpenAI
from llms import read_pdf, load_api_key, compute_price

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.resolve()
GPT_MODEL_NAME = "gpt-3.5-turbo-0125"

API_KEY = load_api_key()

# Creating the client
client = OpenAI(api_key=API_KEY)

# Loading the pdf
paper = read_pdf(ROOT_DIR / "data" / "raw" / "example_pdfs" / "score_based.pdf")

# Setting up the prompts
system_prompt = """
You are BayesGPT, a helpful model that can answer questions about any Machine Learning paper. The user
will provide you a paper, and your goal is to summarize
the paper in bulleted lists, split into three sections:
core argument, evidence, and conclusions.
"""

# TODO: split paper into chunks of 16k tokens.
user_prompt = f"""
The following is a paper on score-based generative modeling. Please summarize the paper in bulleted lists, split into three sections: core argument, evidence, and conclusions.

Paper chunk:

{paper[-40_000:]}
"""

# Querying the model
completion = client.chat.completions.create(
    model=GPT_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": system_prompt.replace("\n", " "),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
)

# Printing the results
print(completion.choices[0].message.content)
print(
    f"input tokens: {completion.usage.prompt_tokens}, "
    f"output tokens: {completion.usage.completion_tokens}\n"
    f"Cost: {compute_price(completion)} USD"
)
```

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
