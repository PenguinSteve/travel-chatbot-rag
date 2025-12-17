import os
import shutil
import uuid
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.storage import MongoDBStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
class DataService:
    REQUIRED_EXCEL_COLUMNS = ['Name', 'Document', 'Location', 'Topic', 'Source']

    @staticmethod
    def ingest_excel(
        file: UploadFile, 
        retriever: ParentDocumentRetriever
    ) -> List[str]:
        try:
            df = pd.read_excel(file.file).fillna("")
        except Exception as e:
            raise ValueError(f"Không thể đọc file Excel. Lỗi: {str(e)}")
        
        # Kiểm tra các cột bắt buộc
        current_columns = df.columns.tolist()
        
        # Tìm các cột bị thiếu
        missing_columns = [col for col in DataService.REQUIRED_EXCEL_COLUMNS if col not in current_columns]
        
        if missing_columns:
            raise ValueError(
                f"File Excel không đúng mẫu quy định. "
                f"Thiếu các cột: {', '.join(missing_columns)}. "
                f"Các cột bắt buộc là: {', '.join(DataService.REQUIRED_EXCEL_COLUMNS)}"
            )

        documents = []

        original_id = str(file.filename) + "_" + str(uuid.uuid4())

        df = df[DataService.REQUIRED_EXCEL_COLUMNS]

        for _, row in df.iterrows():
            content = str(row.get("Document", ""))
            if not content.strip():
                continue

            metadata = {col: str(row.get(col, "")) for col in DataService.REQUIRED_EXCEL_COLUMNS if col != "Document"}
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

            doc.metadata["original_id"] = original_id

        if documents:
            print(f"\n---------------------Ingesting {len(documents)} documents from uploaded Excel file---------------------\n")
            retriever.add_documents(documents)
            return original_id
        
        print(f"\n---------------------Ingested {len(documents)} documents from uploaded Excel file successfully---------------------\n")
        return []
    
    @staticmethod
    def ingest_unstructured_file(
        file: UploadFile, 
        metadata: dict, 
        retriever: ParentDocumentRetriever
    ) -> List[str]:
        # 1. Lưu file tạm thời để Loader đọc
        temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            documents = []

            original_id = str(file.filename) + "_" + str(uuid.uuid4())

            # 2. Chọn Loader phù hợp theo đuôi file
            if file.filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_filename)
                documents = loader.load()
            elif file.filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(temp_filename)
                documents = loader.load()
            elif file.filename.lower().endswith(".txt"):
                loader = TextLoader(temp_filename)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {file.filename}")

            if not documents:
                return []

            # Gộp nội dung (nếu file có nhiều trang) thành 1 Document cha lớn
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Gán metadata từ request
            final_metadata = metadata.copy()

            final_metadata["original_id"] = original_id
            
            new_doc = Document(page_content=full_content, metadata=final_metadata)

            # 4. Thêm vào Retriever
            retriever.add_documents([new_doc])
            return original_id

        finally:
            # Dọn dẹp file tạm
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    @staticmethod
    def delete_document(
        original_id: str, 
        retriever: ParentDocumentRetriever
    ):
        print(f"--- Deleting Document ID: {original_id} ---")
        
        # Xóa trong MongoDB (Docstore)
        try:
            if hasattr(retriever.docstore, "collection"):
                # Query: Xóa tất cả doc mà metadata.original_id == original_id
                result = retriever.docstore.collection.delete_many(
                    {"value.metadata.original_id": original_id}
                )
                print(retriever.docstore.collection)
                print(f"Deleted {result.deleted_count} parent docs from MongoDB")
        except Exception as e:
            print(f"Warning: Failed to delete from MongoDB (ID might not exist): {e}")

        # Xóa trong Pinecone (Vectorstore)
        try:
            # Dùng filter original_id để xóa tất cả chunks con
            retriever.vectorstore.delete(filter={"original_id": original_id})
            print("Successfully deleted from Pinecone Vectorstore")
            return True
        except Exception as e:
            print(f"Error deleting from Pinecone: {e}")
        return False

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