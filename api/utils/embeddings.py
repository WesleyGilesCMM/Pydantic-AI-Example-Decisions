from openai import AzureOpenAI
import os


def get_embedding_client() -> AzureOpenAI:
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
            api_key=os.getenv("EMBEDDINGS_KEY"),
            api_version="2023-05-15",
        )
        return client
    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {e}")
        return None


def embed_text(input: str, model:str = "text-embedding-ada-002") -> list[float]:
    client = get_embedding_client()
    if not client:
        return []
    response = client.embeddings.create(input=input, model=model)
    return response.data[0].embedding
