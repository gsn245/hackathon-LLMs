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
