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
