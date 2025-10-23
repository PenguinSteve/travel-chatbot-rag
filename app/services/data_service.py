from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from typing import List
from langchain_core.documents import Document

class DataService:

    @staticmethod
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