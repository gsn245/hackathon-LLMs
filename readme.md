# Starting kit - LLMs

Some loose ideas:
1. reviews papers (maybe using supervised information from OpenReview). Aim: comparing with the NeurIPS experiment.
2. answers the question "has this idea been done already?" (verify with one's own previous work)
3. given a methods section, writes the related work section.
4. Study the bias and ethical concerns that arise from these tasks.

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

## Adding your credentials

Ask Miguel for `the-key`, and add your OpenReview credentials to it. Leave the file on the root directory, right next to this readme.

## Basic examples

You can find these examples inside `examples/basic` in this repository.

- [A simple query to GPT 3.5 or 4](#a-simple-query-to-gpt-354)
- [Reading a pdf](#reading-a-pdf)
- [Counting tokens](#counting-tokens)
- [Passing a pdf to GPT](#passing-a-pdf-to-gpt-354-and-asking-for-a-summary)
- [Downloading pdfs from OpenReview](#downloading-pdfs-from-a-conference-in-openreview)
- [Downloading reviews from OpenReview](#downloading-reviews-from-openreview)
- [Computing embeddings](#computing-embeddings-of-abstracts-from-arxiv)


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

### Counting tokens

[🧑‍🍳 Here's an example from the OpenAI cookbook.](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)

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

```python
from pathlib import Path

# Remember to add your username and password to the-key
from llms import load_openreview_credentials

import openreview

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.resolve()

username, password = load_openreview_credentials()

# API V2
client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=username,
    password=password,
)

# Get the venue id from the URL of the conference
notes = client.get_all_notes(
    content={"venueid": "ICLR.cc/2024/Conference"}, details="original"
)

# Print the title of each paper
for note in notes:
    print(note.id, note.content["title"])

# Downloading the first 10 papers
SAVE_PATH = ROOT_DIR / "data" / "raw" / "downloaded_from_openreview"
SAVE_PATH.mkdir(parents=True, exist_ok=True)
for note in notes[:10]:
    pdf = client.get_pdf(note.id)
    with open(SAVE_PATH / f"{note.id}.pdf", "wb") as f:
        f.write(pdf)

```

### Downloading reviews from OpenReview

```python
from pathlib import Path
from llms import load_openreview_credentials

import openreview

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.resolve()

# Loading the credentials
username, password = load_openreview_credentials()

# Creating a client for OpenReview
client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=username,
    password=password,
)

# Getting all submissions to the conference.
venue_id = "ICLR.cc/2024/Conference"
venue_group = client.get_group(venue_id)
submission_name = venue_group.content["submission_name"]["value"]

# Reference for the schema of submissions:
# https://docs.openreview.net/reference/api-v2/openapi-definition#notes
submissions = client.get_all_notes(
    invitation=f"{venue_id}/-/{submission_name}", details="replies"
)

review_name = venue_group.content["review_name"]["value"]

# Printing the reviews for the first 10 submissions.
for submission in submissions[:10]:
    print("-" * 50)
    print(submission.id)
    print(submission.content["title"]["value"])

    for reply in submission.details["replies"]:
        if (
            f"{venue_id}/{submission_name}{submission.number}/-/{review_name}"
            in reply["invitations"]
        ):
            # Printing the rating
            print(reply["content"]["rating"]["value"])

            # Printing the confidence
            # print(reply["content"]["confidence"]["value"])

```

### Downloading papers from arXiv

```python
from pathlib import Path

import arxiv

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
SAVE_DIR = ROOT_DIR / "data" / "raw" / "downloaded_from_arxiv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Construct the default API client.
# The terms of service from arXiv specify that
# we need to have a delay of 3 seconds between
# API calls. arxiv.py handles this for us.
client = arxiv.Client()

# Search for the 5 most recent articles matching the keyword "Bayesian optimization."
search = arxiv.Search(
    query="Bayesian optimization",
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

results = client.results(search)

for result in results:
    print("-" * 50)
    print(result.entry_id + "\n")
    print(result.title + "\n")
    print(result.summary + "\n")
    print(result.pdf_url)

    result.download_pdf(dirpath=SAVE_DIR)

```

### Computing embeddings of abstracts from arxiv

```python
import json
from pathlib import Path

from openai import OpenAI

import arxiv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llms import load_api_key

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
SAVE_DIR = ROOT_DIR / "data" / "raw" / "downloaded_from_arxiv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = load_api_key()

client_arxiv = arxiv.Client()

# Extracting abstracts from three different topics
# on arXiv
abstracts = []
for query in ["Bayesian Optimization", "Variational inference", "score-based model"]:
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = client_arxiv.results(search)
    abstracts_ = [result.summary.replace("\n", " ") for result in results]

    abstracts.extend(abstracts_)

client_open_ai = OpenAI(api_key=API_KEY)

# API reference for embeddings:
# https://beta.openai.com/docs/api-reference/embeddings
abstract_embeddings = [
    client_open_ai.embeddings.create(input=[abstract], model="text-embedding-3-small")
    .data[0]
    .embedding
    for abstract in abstracts
]

print(abstract_embeddings)

with open(SAVE_DIR / "abstracts_and_embeddings.json", "w") as fp:
    json.dump(
        {"abstracts": abstracts, "abstract_embeddings": abstract_embeddings},
        fp,
    )

# Searching for a relevant abstract given a small prompt
prompt = "Bayesian Optimization"

# Computing the embedding of the prompt
prompt_embedding = (
    client_open_ai.embeddings.create(input=[prompt], model="text-embedding-3-small")
    .data[0]
    .embedding
)

# Computing the similarity between the prompt and the abstracts
prompt_embedding = np.array(prompt_embedding)
abstract_embeddings = np.array(abstract_embeddings)

similarities = [
    cosine_similarity(
        prompt_embedding.reshape(1, -1), embedding.reshape(1, -1)
    ).flatten()[0]
    for embedding in abstract_embeddings
]

print(similarities)
print(abstracts[np.argmax(similarities)])
```
