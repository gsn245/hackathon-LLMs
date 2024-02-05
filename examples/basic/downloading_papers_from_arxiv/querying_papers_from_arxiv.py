"""A minimal example of how to use arxiv.py

Taken from the readme of the package:
https://github.com/lukasschwab/arxiv.py
"""
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
