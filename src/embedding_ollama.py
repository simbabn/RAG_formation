from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_ollama_embeddings():
    """
    Initialize and return the Ollama embeddings instance.
    This function can be customized to use different models or configurations.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"  # Adjust the URL as needed
    )