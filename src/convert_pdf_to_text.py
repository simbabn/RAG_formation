#from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document as Documents

from embedding_ollama import get_ollama_embeddings
from langchain.vectorstores.chroma import Chroma 

CHROMA_PATH = "chroma"
DATA_PATH = "../data/test/"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Documents]):
    """
    Split the documents into smaller chunks for better processing.
    This function can be customized to use different text splitters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjust chunk size as needed
        chunk_overlap=80,  # Adjust overlap as needed
        length_function=len,  # Use the default length function
        is_separator_regex=False  # Use a simple separator
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Documents]):
    """
    Add the chunks to a vector store (e.g., ChromaDB) for later retrieval.
    This function can be customized to use different vector stores.
    """

    db = Chroma(
        persist_directory="../data/chroma_db",
        embedding_function=get_ollama_embeddings(),
    )
    
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    #Add or Update the documents 
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing items in the database: {len(existing_ids)}")
    
    #add document that are not in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new chunks to the database.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# documents = load_documents()
# chunks = split_documents(documents)
# print(chunks[1000])  # Print the first 1000 characters of the first document




if __name__ == "__main__":
    main()