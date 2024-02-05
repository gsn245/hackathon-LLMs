from openai.types.completion import Completion

from llms.utils.constants import MODELS_AND_PRICES


def compute_price(chat_completion: Completion) -> float:
    """
    Returns the price of the interaction with the chat on USD.
    """
    model_name = chat_completion.model
    input_tokens = chat_completion.usage.prompt_tokens
    output_tokens = chat_completion.usage.completion_tokens

    return (
        MODELS_AND_PRICES[model_name]["input"] * input_tokens
        + MODELS_AND_PRICES[model_name]["output"] * output_tokens
    )
