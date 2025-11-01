from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from app.config.settings import settings
from app.core.llm import llm_chat
import os
from app.models.chat_schema import ChatMessage
from app.repositories.chat_repository import ChatRepository
from app.request.AskRequest import ChatRequest

GROQ_API_KEY = settings.GROQ_API_KEY
LLM_MODEL = settings.LLM_MODEL

class RAGService:
    
    @staticmethod
    def generate_groq_response(retriever, payload: ChatRequest, standalone_question: str, chat_history: list, topic: str = None, location: str = None, chat_repository: ChatRepository = None):
        try:
            message = payload.message
            session_id = payload.session_id

            llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)

            system = """You are an AI assistant that helps people find information about tourism.
            You are given the following: conversation so far, extracted parts of a long document and a question.
            Provide a conversational answer based on the context and conversation provided.
            If the question is about greeting or is off-topic, respond appropriately.
            If you don't know the answer or the context doesn't contain relevant information, just say "Hiện tại tôi không thể trả lời câu hỏi của bạn vì tôi thiếu thông tin về dữ liệu đó".
            Always answer in Vietnamese.
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", "Conversation:\n{conversation}\n\nContext:\n{context}\n\nQuestion: {question}")
            ])

            rag_chain = prompt | llm | StrOutputParser()

            conversation_lines = []

            for msg in chat_history:
                if msg.type == 'human':
                    conversation_lines.append(f"Người dùng: {msg.content}")
                elif msg.type == 'ai':
                    conversation_lines.append(f"Trợ lý: {msg.content}")

            conversation_str = "\n".join(conversation_lines)

            if( topic == "Off_topic" ):
                prompt_input = {
                    "conversation": conversation_str,
                    "context": "",
                    "question": standalone_question
                }
                

                response = rag_chain.invoke(prompt_input)

                chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
                chat_repository.save_message(session_id=session_id, message=ChatMessage(content=response, role="ai"))
                return response, []
            
            
            
            # Retrieve relevant documents
            context_docs = RAGService.retrieve_documents(retriever, standalone_question)

            prompt_input = {
                "conversation": conversation_str,
                "context": "\n\n".join([doc.page_content for doc in context_docs]),
                "question": standalone_question
            }

            response = rag_chain.invoke(prompt_input)

            chat_repository.save_message(session_id=session_id, message=ChatMessage(content=message, role="human"))
            chat_repository.save_message(session_id=session_id, message=ChatMessage(content=response, role="ai"))
            
            print("\n---------------------Generated response:---------------------\n")
            print(response)
            print("\n---------------------Context documents:---------------------\n")
            return response, context_docs

        except Exception as e:
            raise RuntimeError(f"RAG generation error: {e}")
        
    @staticmethod
    def classify_query(query: str):
        try:
            llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)

            system = """You are a classifier assistant. Based on the user's question, extract the 'topic' and 'location'.
            The 'topic' must be one of: ['Food', 'Accommodation', 'Attraction', 'General', 'Festival', 'Restaurant', 'Transport', 'Off_topic', 'Plan'].
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
            {{"Topic": "Transport", "Location": "Hà Nội"}}

            Example 6: "Tell me a joke."
            {{"Topic": "Off_topic", "Location": null}}

            Example 7: "Xin chào!"
            {{"Topic": "Off_topic", "Location": null}}
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

    @staticmethod
    def retrieve_documents(retriever, query: str):
        try:
            start_time_retrieval = os.times()
            print("\n---------------------Retrieving relevant documents...---------------------\n")
            context_docs = retriever.invoke(query)
            end_time_retrieval = os.times()
            print("\n---------------------Retrieved relevant documents in", end_time_retrieval.user - start_time_retrieval.user, "seconds---------------------\n")
            
            return context_docs

        except Exception as e:
            raise RuntimeError(f"Document retrieval error: {e}")

    @staticmethod
    def build_standalone_question(question: str, chat_history: list):
        
        contextualize_q_system_prompt = """Bạn là một trợ lý AI chuyên viết lại câu hỏi. Nhiệm vụ duy nhất của bạn là lấy Lịch sử trò chuyện và Câu hỏi mới, sau đó tạo ra một "Câu hỏi độc lập" (standalone question) duy nhất có thể hiểu được mà không cần lịch sử.

            QUY TẮC TUYỆT ĐỐI:
            - KHÔNG BAO GIỜ được trả lời câu hỏi.
            - CHỈ được xuất ra (output) câu hỏi độc lập đã được viết lại.
            - Nếu câu hỏi mới đã đủ nghĩa, hãy lặp lại y hệt.
            - Không thêm bất kỳ lời chào hay lời giải thích nào.
            - Phải giữ nguyên đại từ của người dùng nếu không có đại từ thì sử dụng đại từ "tôi" (ví dụ: "tôi", "cho tôi", "của tôi"). KHÔNG được đổi thành "bạn".

            VÍ DỤ:
            ---
            Lịch sử: [Human: "Tôi muốn đi du lịch Đà Nẵng"]
            Câu hỏi mới: "Ở đó có gì chơi?"
            Câu hỏi độc lập: "Đà Nẵng có những địa điểm du lịch nào?"
            ---
            Lịch sử: [Human: "Cầu Rồng đẹp thật!", AI: "Đúng vậy, Cầu Rồng phun lửa vào cuối tuần."]
            Câu hỏi mới: "Mấy giờ vậy?"
            Câu hỏi độc lập: "Cầu Rồng phun lửa lúc mấy giờ vào cuối tuần?"
            ---
            Lịch sử: []
            Câu hỏi mới: "Các món ăn ngon ở Hà Nội là gì?"
            Câu hỏi độc lập: "Các món ăn ngon ở Hà Nội là gì?"
            ---
            Lịch sử: [Human: "Các món ăn ở Đà Nẵng là gì?", AI: "Đà Nẵng có Mì Quảng, Bánh Xèo..."]
            Câu hỏi mới: "Ngoài những món đó ra, còn món nào khác không?"
            Câu hỏi độc lập: "Ngoài Mì Quảng và Bánh Xèo, Đà Nẵng còn có những món ăn nào khác ở Đà Nẵng?"
            ---

            Bây giờ, hãy thực hiện nhiệm vụ cho Lịch sử và Câu hỏi mới dưới đây:
            """

        history_lines = []
        for msg in chat_history:
            role = "Human" if msg.type == 'human' else "AI"
            history_lines.append(f"{role}: \"{msg.content}\"")
        
        chat_history_str = ", ".join(history_lines)
        if chat_history_str:
            chat_history_str = f"[{chat_history_str}]"
        else:
            chat_history_str = "[]"

        print('---> Formatted Chat History:', chat_history_str)

        # Create prompt template for contextualizing question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("human", "Lịch sử: {chat_history}\nCâu hỏi mới: {input}")
            ]
        )

        # Create the chain for generating standalone question
        contextualize_q_chain = contextualize_q_prompt | llm_chat() | StrOutputParser()

        standalone_question = contextualize_q_chain.invoke(
            {
                "input": question,
                "chat_history": chat_history_str
            }
        )

        return standalone_question