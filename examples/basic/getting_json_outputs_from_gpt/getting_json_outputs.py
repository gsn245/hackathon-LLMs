"""
https://platform.openai.com/docs/guides/text-generation/json-mode
"""

from openai import OpenAI

from llms import load_api_key, compute_price

GPT_MODEL_NAME = "gpt-3.5-turbo-0125"
API_KEY = load_api_key()

client = OpenAI(api_key=API_KEY)

prompt = "Give me three good papers to get started on Bayesian Optimization. Structure your output as a JSON object with the following scheme: {'papers': [{'title': '...', 'authors': ['...']}]"

response = client.chat.completions.create(
    model=GPT_MODEL_NAME,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are BayesGPT, a helpful assistant who knows everything there is to know about probabilistic Machine Learning. You output everything as a JSON object.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
)
print(response.choices[0].message.content)
print(
    f"input tokens: {response.usage.prompt_tokens}, "
    f"output tokens: {response.usage.completion_tokens}\n"
    f"Cost: {compute_price(response)} USD"
)
