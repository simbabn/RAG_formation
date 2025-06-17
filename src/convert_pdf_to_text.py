#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_documents():
    """
    Load all PDF documents from the 'data' directory and return them as a list of documents.
    Each document is represented as a string containing the text extracted from the PDF.
    """
    document_loader = PyPDFDirectoryLoader("../data", glob="larousse-des-plantes-medicinales.pdf")
    return document_loader.load()



documents = load_documents()
print(documents[100].page_content[:1005])  # Print the first 1000 characters of the first document