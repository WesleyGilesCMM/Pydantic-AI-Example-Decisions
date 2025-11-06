import os
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

__all__ = ["get_llm"]


def get_azure_client() -> AsyncAzureOpenAI:
    if not all(x in os.environ for x in ("AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT")):
        raise ValueError(
            "Using Azure OpenAI models requires the env variables `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT`"
        )
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-10-01-preview"),
    )
    return client


def get_llm(model_string: str) -> OpenAIChatModel | str:
    """Helper function to handle using Azure OpenAI resources in Pydantic AI"""
    if model_string.startswith("azure_openai"):
        client = get_azure_client()
        return OpenAIChatModel(
            model_name=model_string.split(":")[-1],
            provider=OpenAIProvider(openai_client=client),
        )
    return model_string
