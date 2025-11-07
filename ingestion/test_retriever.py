from langchain_community.document_compressors import FlashrankRerank
from langchain_community.storage import MongoDBStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.vector_database_pinecone import PineconeConfig
from app.config.settings import settings
from app.services.rag_service import RAGService
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from evaluate.evaluate_rag import RAGEvaluation

def main():
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"

    flashrank_compressor = FlashrankRerank(top_n=2, model="ms-marco-MiniLM-L-12-v2")

    docstore = MongoDBStore(
        connection_string=CONNECTION_STRING,
        db_name=settings.MONGO_DB_NAME,
        collection_name=settings.MONGO_STORE_COLLECTION_NAME
    )

    vector_store = PineconeConfig().get_vector_store()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""])
    
    question = "Giao thông ở TP.HCM khác gì so với Hà Nội?"

    classify_result = RAGService.classify_query(query=question)

    topic = classify_result.get("Topic", "None")
    location = None

    filter = {}
    if topic:
        filter["Topic"] = topic
    if location:
        filter["Location"] = location

    base_retriever_pdr = ParentDocumentRetriever(
        child_splitter=child_splitter,
        docstore=docstore,
        vectorstore=vector_store,
        search_kwargs={
            "k": 10,
            "filter": filter
        }
    )

    base_retriever_pdr.search_kwargs["filter"] = filter

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever_pdr,
        base_compressor=flashrank_compressor
    )

    if topic != 'Plan':
        rag_evaluation = RAGEvaluation()

        response, context = rag_evaluation.generate_response(compression_retriever, question, topic, location)
        print("\n---------------------Generated Response---------------------\n")
        print("Response:", response)
        print("\n---------------------Context Documents---------------------\n")
        for i, doc in enumerate(context):
            print(f"Document {i+1}:", doc.page_content)
            print("Metadata:", doc.metadata)
            print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()