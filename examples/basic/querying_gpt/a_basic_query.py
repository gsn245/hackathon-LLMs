from openai import OpenAI

from llms.api import load_api_key
from llms.utils.constants import MODELS_AND_PRICES

API_KEY = load_api_key()

# The options for the GPT model:

# Older model, 4k token window.
# GPT_MODEL_NAME = "gpt-3.5-turbo-0613"

# Has JSON mode, can do parallel function calling...
# GPT_MODEL_NAME = "gpt-3.5-turbo-1106"

# Recommended model: Long context windows (16k tokens).
# (and cheapest).
GPT_MODEL_NAME = "gpt-3.5-turbo-0125"

client = OpenAI(api_key=API_KEY)

completion = client.chat.completions.create(
    model=GPT_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are BayesGPT, a helpful assistant who is an expert on probabilistic modeling.",
        },
        {
            "role": "user",
            "content": "What is an intuitive description of the Kullback-Leibler divergence?",
        },
    ],
)

input_tokens = completion.usage.prompt_tokens
output_tokens = completion.usage.completion_tokens

print(completion.choices[0].message.content)
print("\n")
print(
    f"Cost: {MODELS_AND_PRICES[GPT_MODEL_NAME]['input'] * input_tokens + MODELS_AND_PRICES[GPT_MODEL_NAME]['output'] * output_tokens} USD"
)
