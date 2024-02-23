"""
In case you start running into rate limits with the OpenAI API, you can use the library Tenacity to retry the request with a delay. Here's an example of how to do that.

Taken from:
https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
"""

from openai import OpenAI

from llms import load_api_key, compute_price


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

API_KEY = load_api_key()
client = OpenAI(api_key=API_KEY)

GPT_MODEL_NAME = "gpt-4"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


completion = completion_with_backoff(
    model=GPT_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are BayesGPT, a helpful chat model and a know-it-all of probabilistic modeling.",
        },
        {
            "role": "user",
            "content": "What is the marginal log-likelihood and why is it important?",
        },
    ],
)

print(completion.choices[0].message.content)
print(
    f"input tokens: {completion.usage.prompt_tokens}, "
    f"output tokens: {completion.usage.completion_tokens}\n"
    f"Cost: {compute_price(completion)} USD"
)
