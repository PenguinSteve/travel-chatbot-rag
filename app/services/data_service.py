from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.storage import MongoDBStore
from langchain.retrievers import ParentDocumentRetriever

class DataService:

    @staticmethod
    def ingest_data(documents: List[Document], store: MongoDBStore, vector_store: PineconeVectorStore):
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""])
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        parent_document_retriever = ParentDocumentRetriever(
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            docstore=store,
            vectorstore=vector_store
        )

        print(f"\n---------------------Ingesting {len(documents)} documents into MongoDB store and Pinecone vector store---------------------\n")
        parent_document_retriever.add_documents(documents)
        print(f"\n---------------------Ingested {len(documents)} documents successfully---------------------\n")


    @staticmethod
    def load_raw_data(
        filepath: str,
        document_column: str = "Document",
        metadata_columns: List[str] = ["Name", "Location", "Topic", "Source"]
    ) -> List[Document]:
        df = pd.read_excel(filepath).fillna("")

        documents = []

        # Load documents and metadata
        for index, row in df.iterrows():
            main_document_text = str(row.get(document_column, ""))

            metadata = {col: str(row.get(col, "")) for col in metadata_columns}
            
            doc = Document(page_content=main_document_text, metadata=metadata)

            documents.append(doc)

        print(f"\n---------------------Transformed {len(df)} rows into {len(documents)} document chunks---------------------\n")
        return documents