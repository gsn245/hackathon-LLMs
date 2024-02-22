import tiktoken

# Best model.
# GPT_MODEL_NAME = "gpt-4"  # Or "gpt-4-32k" for longer context windows.

# Older model, 4k token window.
# GPT_MODEL_NAME = "gpt-3.5-turbo-0613"

# Has JSON mode, can do parallel function calling...
# GPT_MODEL_NAME = "gpt-3.5-turbo-1106"

# Recommended model (cheapest, 16k tokens).
GPT_MODEL_NAME = "gpt-3.5-turbo-0125"

# Getting the encoding for a given model:
encoding = tiktoken.encoding_for_model(GPT_MODEL_NAME)

# Printing the encoding:
sentence = "I am BayesGPT."
tokens = encoding.encode(sentence)
print(f"{sentence} -> {tokens} ({len(tokens)} tokens)")

# Decoding the tokens:
decoded_sentence = encoding.decode(tokens)
print(f"{tokens} -> {decoded_sentence}")
