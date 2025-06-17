import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding_ollama import get_ollama_embeddings


PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.`
{context}

----
Answer the question based on the context provided above.
Question: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    emveddings = get_ollama_embeddings()
    db = Chroma(
        persist_directory="../data/chroma_db",
        embedding_function=emveddings
    )
    
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    
    context = "\n".join([f"Score: {score}\n{doc.page_content}" for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context,
        question=query_text
    )
    
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print("Prompt : ",prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
    