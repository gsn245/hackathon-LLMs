from openai import OpenAI

from llms.api import load_api_key, compute_price

API_KEY = load_api_key()

# The options for the GPT model:

# Best model.
# GPT_MODEL_NAME = "gpt-4"  # Or "gpt-4-32k" for longer context windows.

# Older model, 4k token window.
# GPT_MODEL_NAME = "gpt-3.5-turbo-0613"

# Has JSON mode, can do parallel function calling...
# GPT_MODEL_NAME = "gpt-3.5-turbo-1106"

# Recommended model (cheapest, 16k tokens).
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

print(completion.choices[0].message.content)
print(
    f"input tokens: {completion.usage.prompt_tokens}, "
    f"output tokens: {completion.usage.completion_tokens})"
    "\n"
    f"Cost: {compute_price(completion)} USD"
)
