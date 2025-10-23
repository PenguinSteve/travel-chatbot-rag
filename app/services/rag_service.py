from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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
        
    @staticmethod
    def classify_query(query: str):
        try:
            llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)

            system = """You are a classifier assistant. Based on the user's question, extract the 'topic' and 'location'.
            The 'topic' must be one of: ['Food', 'Accommodation', 'Attraction', 'General', 'Festival', 'Restaurant', 'Transport', 'Plan'].
            The 'location' must be one of: ['Hà Nội', 'Thành phố Hồ Chí Minh', 'Đà Nẵng'].
            If a value is not mentioned, return null for that key.
            Respond ONLY with a valid JSON object.

            Example 1: "Quán phở nào ngon ở Hà Nội?"
            {{"Topic": "Food", "Location": "Hà Nội"}}

            Example 2: "Khách sạn nào tốt?"
            {{"Topic": "Accommodation", "Location": null}}
            
            Example 3: "Thời gian tốt để thăm Đà Nẵng là khi nào?"
            {{"Topic": "General", "Location": "Đà Nẵng"}}
            
            Example 4: "Tôi muốn biết về các lễ hội ở Thành phố Hồ Chí Minh."
            {{"Topic": "Festival", "Location": "Thành phố Hồ Chí Minh"}}

            Example 5: "Các cách di chuyển ở Hà Nội"
            {{"Topic": "Transport", "Location": null}}
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", "Question: {question}")
            ])

            classification_chain = prompt | llm | JsonOutputParser()

            print(f"\n---------------------Classifying query: {query}---------------------\n")
            classification = classification_chain.invoke({"question": query})
            print(f"\n---------------------Classification result: {classification}---------------------\n")
            return classification

        except Exception as e:
            raise RuntimeError(f"Query classification error: {e}")