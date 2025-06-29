from langchain_community.embeddings import BedrockEmbeddings


def get_bedrock_embeddings():
    """
    Initialize and return the Bedrock embeddings instance.
    This function can be customized to use different models or configurations.
    """
    return BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v1"
    )