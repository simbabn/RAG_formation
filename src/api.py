from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

from fastapi.middleware.cors import CORSMiddleware


from src.embedding_ollama import get_ollama_embeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 Pendant le dev, on autorise tout. Tu peux restreindre plus tard.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

embeddings = get_ollama_embeddings()
db = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)
model = Ollama(model="llama3")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(req: QueryRequest):
    query_text = req.query
    results = db.similarity_search_with_score(query_text, k=5)
    context = "\n".join([f"Score: {score}\n{doc.page_content}" for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    return {
        "response": response_text,
        "sources": sources
    }
