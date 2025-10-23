from langchain_groq import ChatGroq
from store_data import connect_to_pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config.settings import settings
import os

GROQ_API_KEY = settings.GROQ_API_KEY
LLM_MODEL = settings.LLM_MODEL

class RAGService:
    
    @staticmethod
    def generate_groq_response(retriever, query: str):
        try:
            llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)

            system = """You are an AI assistant that helps people find information about tourism.
            You are given the following extracted parts of a long document and a question.
            Provide a conversational answer based on the context provided.
            If you don't know the answer or the context doesn't contain relevant information, just say "Hiện tại tôi không thể trả lời câu hỏi của bạn vì tôi thiếu thông tin về dữ liệu đó". Don't try to make up an answer.
            Always answer in Vietnamese.
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", "Context:\n{context}\n\nQuestion: {question}")
            ])
            
            start_time_retrieval = os.times()
            print("\n---------------------Retrieving relevant documents...---------------------\n")
            context_docs = retriever.invoke(query)
            end_time_retrieval = os.times()
            print("\n---------------------Retrieved relevant documents in", end_time_retrieval.user - start_time_retrieval.user, "seconds---------------------\n")

            prompt_input = {
                "context": "\n\n".join([doc.page_content for doc in context_docs]),
                "question": query
            }

            rag_chain = prompt | llm | StrOutputParser()

            response = rag_chain.invoke(prompt_input)
            return response, context_docs

        except Exception as e:
            raise RuntimeError(f"RAG generation error: {e}")