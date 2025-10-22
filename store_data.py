from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_dotenv()

def init_pinecone(index_name: str, dimension: int = 768):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("\n---------------------Created new Pinecone index---------------------\n")
    else:
        print(f"\n---------------------Found existing Pinecone index '{index_name}'---------------------\n")

    return pc.Index(index_name)


def connect_to_pinecone(index_name: str) -> PineconeVectorStore:
    index = init_pinecone(index_name=index_name)
    embedding_model = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    print("\n---------------------Connected to Pinecone vector store----------------------\n")
    return vector_store


def transform_data(
    filepath: str,
    document_column: str = "Document",
    metadata_columns: List[str] = ["Name", "Location", "Topic", "Source"]
) -> List[Document]:
    df = pd.read_excel(filepath).fillna("")

    # Initialize text splitter for document chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    # Chunk documents and preserve metadata
    for index, row in df.iterrows():
        main_document_text = str(row.get(document_column, ""))

        original_metadata = {col: str(row.get(col, "")) for col in metadata_columns}
        chunks = text_splitter.split_text(main_document_text)

        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(page_content=chunk_text, metadata=original_metadata)
            all_chunks.append(chunk_doc)
    
    print(f"\n---------------------Transformed {len(df)} rows into {len(all_chunks)} document chunks---------------------\n")
    return all_chunks
        

def import_data_to_pinecone(chunks: List[Document], index_name: str):
    try:
        vector_store = connect_to_pinecone(index_name)
        print(f"\n---------------------Importing {len(chunks)} chunks to Pinecone index '{index_name}'---------------------\n")
        start_time_import = os.times()
        vector_store.add_documents(documents=chunks)
        end_time_import = os.times()
        print("\n---------------------Data import completed in", end_time_import.user - start_time_import.user, "seconds---------------------\n")
        
    except Exception as e:
        print(f"\n---------------------An error occurred while importing data to Pinecone: {e}---------------------\n")


def main():
    EXCEL_FILE_PATH = "" 
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-tourism")

    init_pinecone(index_name=PINECONE_INDEX_NAME)

    document_chunks = transform_data(filepath=EXCEL_FILE_PATH)

    import_data_to_pinecone(
        chunks=document_chunks, 
        index_name=PINECONE_INDEX_NAME
    )

if __name__ == "__main__":
    # main()
    pass