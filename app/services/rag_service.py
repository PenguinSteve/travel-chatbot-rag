from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from app.core.llm import llm_create_standalone_question, llm_rag, llm_classify
import os
from app.models.chat_schema import ChatMessage
from app.repositories.chat_repository import ChatRepository
from app.request.AskRequest import ChatRequest

class RAGService:
    
    @staticmethod
    def generate_response(retriever, payload: ChatRequest, standalone_question: str, chat_history: list, topic: str = None, location: str = None, chat_repository: ChatRepository = None):
        try:
            message = payload.message
            session_id = payload.session_id

            llm = llm_rag()

            system = """### Role and Goal
                You are an AI assistant specializing in tourism. Your persona is friendly, helpful, and **extremely accurate**.
                Your task is to answer user questions, but you operate under one ABSOLUTE constraint: You MUST ONLY use the information provided to you.

                ### The Golden Rules (MOST IMPORTANT)
                1.  **Strict Faithfulness:** Your answer MUST be **ENTIRELY** derived from the provided 'Context'.
                2.  **No External Knowledge:** You are STRICTLY PROHIBITED from using any external knowledge (your pre-trained knowledge) to answer. If the information is not in the 'Context', you CANNOT say it.
                3.  **Natural Phrasing (No Meta-Talk):**
                * You must sound like a natural, human expert. 
                * **DO NOT** talk about yourself as an AI or mention your data sources. 
                * **AVOID ALL** phrases like: "in the documents I received," "based on the context," "in the provided information," "trong các tài liệu," "dựa trên ngữ cảnh," or "thông tin tôi nhận được."
                * Just state the information directly.
                4.  **Handling **Completely** Missing Information (The "I don't know" rule):**
                * This rule ONLY applies if the 'Context' is **completely empty** OR **contains no relevant information AT ALL** to the 'Question'.
                * In this specific case, you **MUST** respond with this exact Vietnamese phrase: "Hiện tại tôi không thể trả lời câu hỏi của bạn vì tôi thiếu thông tin về dữ liệu đó". Do not add any other explanation.
                5.  **Handling **Partial** Information (Best-Effort Rule):**
                * Your main goal is to be helpful.
                * If the user asks for a specific quantity (e.g., "top 50 dishes"), but the 'Context' provides **fewer items** than requested, you **MUST** provide **all the relevant items you found in the 'Context'**.
                * If this is a follow-up question (e.g., user asks for 70 after you just gave 50), simply state naturally that you don't have additional items.
                * **Example of a good response (natural):** "Hiện tại tôi chỉ có danh sách 50 món ăn này thôi." or "Danh sách của tôi có 50 món, tôi không tìm thấy món nào khác."
                * **Example of a bad response (robot):** "Trong tài liệu tôi chỉ tìm thấy 50 món."
                6.  **Handling Conversation History:**
                    * Use the 'Conversation History' to understand follow-up questions (e.g., "what else?", "besides those...").
                    * When answering a follow-up, **AVOID REPEATING** information already present in the 'Conversation History'. Prioritize NEW information found in the 'Context'.
                7.  **Handling Off-topic/Greeting:** If the 'Question' is a greeting or unrelated to tourism, respond politely, be friendly, and steer the conversation back to tourism (e.g., "Hello, how can I help you with your travel plans today?").
                8. No Post-amble: Do not add any summary sentences at the end explaining where the information came from. Just provide the direct answer.
                9.  **Language:** You must always answer in Vietnamese.
                """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", """Đây là 'Lịch sử trò chuyện' và 'Ngữ cảnh' để bạn tham khảo.
                                    
                    <Lịch sử trò chuyện>
                    {conversation}
                    </Lịch sử trò chuyện>

                    <Ngữ cảnh>
                    {context}
                    </Ngữ cảnh>

                    Hãy trả lời 'Câu hỏi' dưới đây dựa trên các quy tắc đã đề ra.
                    Câu hỏi: {question}""")
            ])

            rag_chain = prompt | llm | StrOutputParser()

            conversation_lines = []

            for msg in chat_history:
                if msg.type == 'human':
                    conversation_lines.append(f"Người dùng: {msg.content}")
                elif msg.type == 'ai':
                    conversation_lines.append(f"Trợ lý: {msg.content}")

            conversation_str = "\n".join(conversation_lines)

            print("\n---------------------Conversation so far:---------------------\n")
            print(conversation_str)

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

            formatted_contexts = []
            for doc in context_docs:
                name = doc.metadata.get('Name', 'Không rõ')
                
                context_str = f"Tên tài liệu: {name}\nNội dung: {doc.page_content}"
                formatted_contexts.append(context_str)

            prompt_input = {
                "conversation": conversation_str,
                "context": "\n\n".join(formatted_contexts),
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
            llm = llm_classify()

            system = """You are a query classifier assistant. Your SOLE task is to analyze the user's "Question" and extract the 'Topic' and 'Location'.

                CLASSIFICATION RULES:

                1.  **`Topic`:** This is the **core intent** of what the user WANTS TO KNOW.
                    * It MUST be one of the following values: `['Food', 'Accommodation', 'Attraction', 'General', 'Festival', 'Restaurant', 'Transport', 'Off_topic', 'Plan']`.
                    * **GOLDEN RULE:** The `Topic` is *what the user is asking about* (e.g., 'Food'), NOT just a named entity in the query (e.g., 'Festival').

                2.  **`Location`:** This is the specific geographical location mentioned.
                    * It MUST be one of the following values: `['Hà Nội', 'Thành phố Hồ Chí Minh', 'Đà Nẵng']`.
                    * If a location is mentioned but is NOT in this list (e.g., "Huế"), return `null`.
                    * If no location is mentioned, return `null`.

                OUTPUT FORMAT:
                -   You MUST respond ONLY with a valid JSON object.
                -   Do NOT provide any explanation or greeting.

                ---
                **EXAMPLE 1 (CRITICAL CASE):**
                Question: "Ẩm thực trong sự kiện Enjoy Danang Festival 2025 tập trung ở khu vực nào?" (Food at the Enjoy Danang Festival 2025)
                Output:
                {{"Topic": "Food", "Location": "Đà Nẵng"}}

                ---
                **EXAMPLE 2 (CONTRASTING CASE):**
                Question: "Thông tin về Enjoy Danang Festival 2025" (Information about Enjoy Danang Festival 2025)
                Output:
                {{"Topic": "Festival", "Location": "Đà Nẵng"}}

                ---
                **EXAMPLE 3:**
                Question: "Khách sạn nào tốt?" (Which hotel is good?)
                Output:
                {{"Topic": "Accommodation", "Location": null}}

                ---
                **EXAMPLE 4:**
                Question: "Món bún bò Huế có ngon không?" (Is Bun Bo Hue good?)
                Output:
                {{"Topic": "Food", "Location": null}}

                ---
                **EXAMPLE 5:**
                Question: "Món bún nêm có gì đặc trưng và quán nào ngon?" (What is special about bun nem and which restaurant is good?)
                Output:
                {{"Topic": "Food", "Location": null}}

                ---
                **EXAMPLE 6:**
                Question: "Quán phở nào ngon ở Hà Nội?" (Which pho restaurant is good in Hanoi?)
                Output:
                {{"Topic": "Food", "Location": "Hà Nội"}}

                ---
                **EXAMPLE 7:**
                Question: "Thời gian tốt để thăm Đà Nẵng là khi nào?" (When is a good time to visit Da Nang?)
                Output:
                {{"Topic": "General", "Location": "Đà Nẵng"}}

                ---
                **EXAMPLE 8:**
                Question: "Tôi muốn biết về các lễ hội ở Thành phố Hồ Chí Minh." (I want to know about festivals in Ho Chi Minh City.)
                Output:
                {{"Topic": "Festival", "Location": "Thành phố Hồ Chí Minh"}}

                ---
                **EXAMPLE 9:**
                Question: "Các cách di chuyển ở Hà Nội" (Ways to get around in Hanoi)
                Output:
                {{"Topic": "Transport", "Location": "Hà Nội"}}

                ---
                **EXAMPLE 10:**
                Question: "Tell me a joke."
                Output:
                {{"Topic": "Off_topic", "Location": null}}

                ---
                **EXAMPLE 11:**
                Question: "Xin chào!" (Hello!)
                Output:
                {{"Topic": "Off_topic", "Location": null}}
                ---

                Now, classify the Question below:
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
        
        contextualize_q_system_prompt = """Bạn là một công cụ viết lại câu. Nhiệm vụ duy nhất của bạn là chuyển đổi "Câu hỏi mới" và "Lịch sử trò chuyện" thành một "Câu hỏi độc lập" (standalone question) có thể hiểu được mà không cần lịch sử.

            QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):

            1.  **NGHIÊM CẤM TRẢ LỜI CÂU HỎI:** Vai trò của bạn KHÔNG phải là trả lời. Nhiệm vụ chỉ là VIẾT LẠI CÂU HỎI. Nếu bạn trả lời, bạn đã thất bại.
            2.  **QUY TẮC "LẶP LẠI" (ƯU TIÊN SỐ 1):** Nếu "Câu hỏi mới" đã là một câu hỏi độc lập, đầy đủ ý nghĩa và không cần lịch sử, BẮT BUỘC phải xuất ra (output) Y HỆT câu hỏi đó.
            3.  **QUY TẮC "VIẾT LẠI" (CHỈ KHI CẦN):** Nếu "Câu hỏi mới" là câu hỏi ngắn, phụ thuộc vào lịch sử (ví dụ: "Ở đó giá bao nhiêu?", "Mấy giờ vậy?"), hãy dùng "Lịch sử trò chuyện" để viết lại thành câu hỏi đầy đủ.
            4.  **GIỚI HẠN OUTPUT:** CHỈ được xuất ra câu hỏi độc lập đã được viết lại. Không thêm lời chào, lời giải thích, hay bất cứ thứ gì khác.
            5.  **GIỮ NGUYÊN ĐẠI TỪ:** Phải giữ nguyên đại từ của người dùng ("tôi", "cho tôi", "của tôi").

            ---
            VÍ DỤ (Làm rõ QUY TẮC "LẶP LẠI"):
            ---
            Lịch sử: []
            Câu hỏi mới: "Các món ăn ngon ở Hà Nội là gì?"
            Câu hỏi độc lập: "Các món ăn ngon ở Hà Nội là gì?"
            ---
            Lịch sử: [Human: "Tôi muốn đi du lịch TPHCM"]
            Câu hỏi mới: "Hãy cho tôi danh sách các món ăn ngon tại hồ chí minh"
            Câu hỏi độc lập: "Hãy cho tôi danh sách các món ăn ngon tại hồ chí minh"
            ---
            Lịch sử: []
            Câu hỏi mới: "Tại quận 4, con đường nào nổi tiếng với các quán hải sản nướng và món ốc đặc sản của TPHCM"
            Câu hỏi độc lập: "Tại quận 4, con đường nào nổi tiếng với các quán hải sản nướng và món ốc đặc sản của TPHCM"
            ---

            VÍ DỤ (Làm rõ QUY TẮC "VIẾT LẠI"):
            ---
            Lịch sử: [Human: "Tôi muốn đi du lịch Huế"]
            Câu hỏi mới: "Ở đó có gì chơi?"
            Câu hỏi độc lập: "Huế có những địa điểm du lịch nào?"
            ---
            Lịch sử: [Human: "Tôi muốn đi Huế"]
            Câu hỏi mới: "lên kế hoạch du lịch 2 ngày"
            Câu hỏi độc lập: "Lên kế hoạch du lịch Huế 2 ngày cho tôi"
            ---
            Lịch sử: [Human: "Cầu Rồng đẹp thật!", AI: "Đúng vậy, Cầu Rồng phun lửa vào cuối tuần."]
            Câu hỏi mới: "Mấy giờ vậy?"
            Câu hỏi độc lập: "Cầu Rồng phun lửa lúc mấy giờ vào cuối tuần?"
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

        print('\n---------------------Formatted Chat History:---------------------\n', chat_history_str)

        # Create prompt template for contextualizing question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("human", "Lịch sử: {chat_history}\nCâu hỏi mới: {input}")
            ]
        )

        # Create the chain for generating standalone question
        contextualize_q_chain = contextualize_q_prompt | llm_create_standalone_question() | StrOutputParser()

        standalone_question = contextualize_q_chain.invoke(
            {
                "input": question,
                "chat_history": chat_history_str
            }
        )

        return standalone_question