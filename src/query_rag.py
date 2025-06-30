import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding_ollama import get_ollama_embeddings


PROMPT_TEMPLATE = """
Tu es un assistant expert et bienveillant, spécialisé en naturopathie. 
Ta mission est d'apporter des réponses détaillées, précises et fiables, en t'appuyant uniquement sur le contexte fourni ci-dessous.

Utilise exclusivement les éléments du contexte pour répondre à la question finale. 
Si l'information n'est pas présente ou pas claire, indique-le explicitement plutôt que d'inventer une réponse.

Contexte :
{context}

----
Réponds maintenant à la question ci-dessous en utilisant uniquement les informations du contexte.
Question : {question}

Réponse (en français, complète et précise) :
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
        persist_directory="data/chroma_db",
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
    
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print("Prompt : ",prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
    