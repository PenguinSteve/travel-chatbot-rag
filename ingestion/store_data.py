import os
from typing import List
from langchain_community.storage import MongoDBStore
import pandas as pd
from app.config.settings import settings
from app.config.vector_database_pinecone import PineconeConfig
from app.services.data_service import DataService
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def init_pinecone(index_name: str, dimension: int = 768):
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")

#     pc = Pinecone(api_key=pinecone_api_key)

#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#         print("\n---------------------Created new Pinecone index---------------------\n")
#     else:
#         print(f"\n---------------------Found existing Pinecone index '{index_name}'---------------------\n")

#     return pc.Index(index_name)


# def connect_to_pinecone(index_name: str) -> PineconeVectorStore:
#     index = init_pinecone(index_name=index_name)
#     embedding_model = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
#     vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

#     print("\n---------------------Connected to Pinecone vector store----------------------\n")
#     return vector_store


# def transform_data(
#     filepath: str,
#     document_column: str = "Document",
#     metadata_columns: List[str] = ["Name", "Location", "Topic", "Source"]
# ) -> List[Document]:
#     df = pd.read_excel(filepath).fillna("")

#     # Initialize text splitter for document chunking
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     all_chunks = []

#     # Chunk documents and preserve metadata
#     for index, row in df.iterrows():
#         main_document_text = str(row.get(document_column, ""))

#         if(index == 0):
#             print(main_document_text)

#         original_metadata = {col: str(row.get(col, "")) for col in metadata_columns}
#         chunks = text_splitter.split_text(main_document_text)

#         for i, chunk_text in enumerate(chunks):
#             chunk_doc = Document(page_content=chunk_text, metadata=original_metadata)
#             all_chunks.append(chunk_doc)
    
#     print(f"\n---------------------Transformed {len(df)} rows into {len(all_chunks)} document chunks---------------------\n")
#     return all_chunks
        

# def import_data_to_pinecone(chunks: List[Document], index_name: str):
#     try:
#         vector_store = connect_to_pinecone(index_name)
#         print(f"\n---------------------Importing {len(chunks)} chunks to Pinecone index '{index_name}'---------------------\n")
#         start_time_import = os.times()
#         vector_store.add_documents(documents=chunks)
#         end_time_import = os.times()
#         print("\n---------------------Data import completed in", end_time_import.user - start_time_import.user, "seconds---------------------\n")
        
#     except Exception as e:
#         print(f"\n---------------------An error occurred while importing data to Pinecone: {e}---------------------\n")


def main():
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"
    script_path = os.path.abspath(__file__)

    script_dir = os.path.dirname(script_path)

    project_root = os.path.dirname(script_dir)
    
    file_path = os.path.join(project_root, "data", "data_tourism_TPHCM.xlsx")
    
    docstore = MongoDBStore(
        connection_string=CONNECTION_STRING,
        db_name=settings.MONGO_DB_NAME,
        collection_name=settings.MONGO_STORE_COLLECTION_NAME
    )
    vector_store = PineconeConfig().get_vector_store()

    documents = DataService.load_raw_data(filepath=file_path)

    print(f"\n---------------------Loaded {len(documents)} raw documents---------------------\n")

    DataService.ingest_data(documents, docstore, vector_store)

if __name__ == "__main__":
    main()
    # pass