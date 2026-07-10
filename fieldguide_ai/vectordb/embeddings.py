from collections.abc import Sequence

from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider:
    """Generate embeddings for text batches using the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.model = model
        self.client = client or OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            encoding_format="float",
        )
        print("--------------------------------")
        print("response: ", response)
        print("--------------------------------")
        return [list(item.embedding) for item in response.data]


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    embedding_provider = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = embedding_provider.embed_texts(["Hello, world!"])
    print("embeddings dimensions: ", len(embeddings[0]))
    print("--------------------------------")
    print(embeddings)
    print("--------------------------------")


if __name__ == "__main__":
    main()