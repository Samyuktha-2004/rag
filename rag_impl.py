import os
import shutil
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

CHROMA_PATH = "chroma"
DATA_PATH = "data/"
#API Key here
API_KEY = "***API-KEY***"

def main():
    try:
        generate_data_store()
        while True:
            query = input("Enter your question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            answer = query_database(query)
            print(f"Answer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_data_store():
    documents = load_document()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_document():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]
    print(document.page_content) 
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    if not API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=API_KEY), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def query_database(query: str):
    if not API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

    # Load the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings(openai_api_key=API_KEY))

    # Embed the query and search in the database
    results = db.similarity_search(query, k=5)

    # Use a QA chain to get the answer from the retrieved documents
    qa_chain = load_qa_chain(OpenAI(api_key=API_KEY), chain_type="stuff")
    answer = qa_chain.run(input_documents=results, question=query)

    return answer

if __name__ == "__main__":
    main()
