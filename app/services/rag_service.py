from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_pinecone import PineconeRerank
from app.core.llm import llm_create_standalone_question, llm_rag, llm_classify
import os
from app.models.chat_schema import ChatMessage
from app.repositories.chat_repository import ChatRepository
from app.repositories.redis_chat_repository import RedisChatRepository
from app.request.AskRequest import ChatRequest

class RAGService:
    
    @staticmethod
    def generate_response(retriever, payload: ChatRequest, standalone_question: str, chat_history: list, topics: list = [], location: list = [], chat_repository: (ChatRepository | RedisChatRepository) = None , pinecone_reranker: PineconeRerank = None):
        try:
            message = payload.message
            session_id = payload.session_id

            llm = llm_rag()

            system = """### Role and Goal
                You are an AI assistant specializing in tourism. Your persona is friendly, helpful, **detail** and **extremely accurate**.
                Your task is to answer user questions, but you operate under one ABSOLUTE constraint: You MUST ONLY use the information provided to you.
                Your goal is to provide the **most comprehensive answer possible** by summarizing all relevant information from the 'Context', not just listing items.

                ### The Golden Rules (MOST IMPORTANT)
                1.  **Strict Faithfulness:** Your answer MUST be **ENTIRELY** derived from the provided 'Context'.
                2.  **No External Knowledge:** You are STRICTLY PROHIBITED from using any external knowledge (your pre-trained knowledge) to answer. If the information is not in the 'Context', you CANNOT say it.
                3.  **Natural Phrasing (No Meta-Talk):**
                * You must sound like a natural, human expert. 
                * **DO NOT** talk about yourself as an AI or mention your data sources. 
                * **AVOID ALL** phrases like: "in the documents I received," "based on the context," "in the provided information," "trong các tài liệu," "dựa trên ngữ cảnh," or "thông tin tôi nhận được."
                * Just state the information directly.
                4.  **The "Be Detailed and Helpful" Rule:**
                * Your main goal is to be helpful and comprehensive.
                * When the user asks for information (e.g., "places to visit," "restaurants," "dishes"), you MUST find all relevant items in the 'Context'.
                * **[CRITICAL INSTRUCTION]:** For **EVERY** item you find, you **MUST NOT** just list its name. You **MUST** provide a detailed summary including any descriptions, information (e.g., hours, address, price), or interesting facts about that item available in the 'Context'.
                * **Handling Partial Lists:** If the 'Context' provides fewer items than the user asked for (e.g., user wants "top 50 dishes," but 'Context' only has 10), you **MUST** provide the detailed summaries for all 10 items you found, and then naturally state you don't have more.
                * **Example of a good response (natural):** "I currently have this list of 10 dishes." (Followed by the 10 detailed descriptions) or "My list has 10 dishes; I couldn't find any others."
                * **Example of a bad response (robot):** "In the documents, I only found 10 dishes."
                5.  **[PRIORITY 1] Handling Off-topic, Greeting, or Vague Questions:**
                * This rule must be checked **FIRST**, before searching the context.
                * If the 'Question' is a greeting or unrelated to tourism, respond politely, be friendly, and steer the conversation back to tourism (e.g., "Hello, how can I help you with your travel plans today?", "I can't help with that, but I can assist you with travel information.").
                * If the 'Question' is too vague to be answerable (e.g., "Give me the address," "Give me specific information," "What's the price?"), respond with the Vietnamese phrase: "Tôi có thể giúp gì cho bạn về thông tin du lịch?".  
                6.  **Handling Conversation History:**
                    * Use the 'Conversation History' to understand follow-up questions (e.g., "what else?", "besides those...").
                    * When answering a follow-up, **AVOID REPEATING** information already present in the 'Conversation History'. Prioritize NEW information found in the 'Context'.
                7.  **Handling **Completely** Missing Information (The "I don't know" rule):**
                * This rule ONLY applies if the 'Context' is **completely empty** OR **contains no relevant information AT ALL** to the 'Question'.
                * In this specific case, you **MUST** respond with this exact Vietnamese phrase: "Hiện tại tôi không thể trả lời câu hỏi của bạn vì tôi thiếu thông tin về dữ liệu đó". Do not add any other explanation.
                8. No Post-amble: Do not add any summary sentences at the end explaining where the information came from. Just provide the direct answer.
                9.  **Language:** You must always answer in Vietnamese.
                10. **Formatting (Strict Markdown):** * The output **MUST** be in clean **Markdown** format. 
                * Use **Bold** (`**text**`) for names or key highlights.
                * Use **Bullet points** (`*` or `-`) for lists.
                * **STRICTLY PROHIBITED:** Do NOT use HTML tags like `<br>`, `\n`, `<div>`, or `<span>` inside the text. If the 'Context' contains HTML tags, **REMOVE** or **REPLACE** them with standard punctuation (commas, periods) or new lines.
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

            if( not topics or 'Off_topic' in topics ):
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
            context_docs = RAGService.retrieve_documents(retriever, standalone_question, pinecone_reranker)

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
            return response, context_docs

        except Exception as e:
            raise RuntimeError(f"RAG generation error: {e}")
        
    @staticmethod
    def classify_query(query: str):
        try:
            llm = llm_classify()

            system = """You are a query classifier assistant. Your SOLE task is to analyze the user's "Question" and extract the 'Topic' and 'Location'.

                ### CLASSIFICATION RULES:

                    1. **Topic**
                    - Represents *what the user wants to know about*.
                    - Must be a **list** (even if only one element).
                    - Allowed values: ['Food', 'Accommodation', 'Attraction', 'General', 'Festival', 'Restaurant', 'Transport', 'Plan', 'Off_topic'].
                    - You may include **multiple topics** if the question clearly refers to multiple aspects.
                        - Example: "Phố ẩm thực Hồ Thị Kỷ có món gì ngon và có điểm tham quan nào gần đó?"  
                        → `"Topic": ["Food", "Attraction"]`
                        - Example: "Khu vực Hội quán Nghĩa An trong lễ hội có bán món ăn nào không?"
                        → `"Topic": ["Food", "Festival", "Attraction"]`
                        - Example: "Những nơi lưu trú nào khi tham gia lễ hội Nghinh Ông Cần Giờ?"
                        → `"Topic": ["Accommodation", "Festival", "Attraction"]`
                    - If the question mentions nearly places or proximity → include relevant topics like 'Attraction', 'Accommodation', or 'Restaurant' as applicable.
                    - Only assign 'Plan' if the user explicitly asks to create an itinerary, a schedule, or a multi-day plan (e.g., "2-day plan", "3 days 2 nights", "itinerary for the weekend").
                    - DO NOT assign 'Plan' if the user is simply asking for options or filtering a search by a part of the day (e.g., "in the morning", "in the evening", "at night").
                    - If the question is unrelated to tourism or a greeting or the question not mentions the main points to do something → `"Topic": ["Off_topic"]`. - Example: "Xin chào!", "What is your name?", "Cho địa chỉ chi tiết"

                    2. **Location**
                    - Represents the **primary geographic area(s)** that are the **context** or **main subject** of the user's question.
                    - Must be a **list** (even if only one element).
                    - Allowed values: ['Hà Nội', 'Thành phố Hồ Chí Minh', 'Đà Nẵng'].
                    - You MUST NOT guess the city if it is not **explicitly** mentioned by name.
                    - If the question contains a **district, ward, street name, or any smaller area** (e.g., "Quận 5", "Cần Giờ", "Phú Thọ Hòa", "Nghĩa An") → you MUST **NOT** map it to any city name.
                    
                    - **RULE 2a (Search Context vs. Subject):** If a city name is part of a *subject* (e.g., "bánh mì Hà Nội") but the question is asking for information *within another city* (e.g., "...ở Hồ Chí Minh"), you MUST **only** include the city that is the *search context* ("Thành phố Hồ Chí Minh").
                    
                    - **RULE 2b (Comparisons):** If the question is *comparing* multiple cities (e.g., "Sự khác biệt giữa Hà Nội và Đà Nẵng"), then include all valid cities mentioned.
                    
                    - If no allowed city name is present as the main context, ALWAYS output an empty list [].

                    3. **Output format**
                    - Output ONLY valid JSON.
                    - Structure:
                        ```json
                        {{
                        "Topic": [...],
                        "Location": [...]
                        }}
                        ```
                    - No explanations or additional text.

                    ---
                    ### EXAMPLES (Demonstrating the PLAINNING Rule):
                    Question: "Hãy giúp tôi lên kế hoạch du lịch tại Cần Giờ"
                    Output:
                    {{"Topic": ["Plan"], "Location": []}}

                    Question: "Những nơi vui chơi ở Sài Gòn buổi tối?"
                    Output:
                    {{"Topic": ["Attraction"], "Location": ["Thành phố Hồ Chí Minh"]}}

                    Question: "Gợi ý 1 số nhà hàng cho buổi tối ở Hà Nội"
                    Output:
                    {{"Topic": ["Food", "Restaurant"], "Location": ["Hà Nội"]}}

                    ### EXAMPLES (Demonstrating the STRICT REJECTION RULE):

                    Question: "Lễ hội Nghinh Ông Cần Giờ thường được tổ chức vào thời gian nào hàng năm?"
                    Output:
                    {{"Topic": ["Festival"], "Location": []}}

                    Question: "Nhà lưu niệm Bác Hồ ở số 5 Châu Văn Liêm có vai trò lịch sử gì?"
                    Output:
                    {{"Topic": ["Attraction"], "Location": []}}

                    Question: "Khu vực Hội quán Nghĩa An trong lễ hội có bán món ăn nào không?"
                    Output:
                    {{"Topic": ["Food", "Festival", "Attraction"], "Location": []}}
                    
                    Question: "Địa đạo Phú Thọ Hòa (Quận Tân Phú) mang giá trị gì?"
                    Output:
                    {{"Topic": ["Attraction"], "Location": []}}
                    
                    Question: "Khách sạn nào gần chợ Bến Thành?"
                    Output: 
                    {{"Topic": ["Accommodation"], "Location": []}}
                    
                    Question: "Phố ẩm thực Hồ Thị Kỷ có món gì ngon và có điểm du lịch nào gần đó?"
                    Output:
                    {{"Topic": ["Food", "Attraction"], "Location": []}}
                    
                    Question: "Khi đến Hội quán Vhị Pù (264 Hải Thượng Lãn Ông), du khách có thể thưởng thức món ăn truyền thống nào của người Hoa tại khu vực Chợ Lớn gần đó?"
                    Output:
                    {{"Topic": ["Food", "Attraction"], "Location": []}}

                    ---
                    
                    ### EXAMPLES (Demonstrating Search Context vs. Subject):

                    Question: "Bánh mì Hà Nội được bán ở đâu ở Hồ Chí Minh?"
                    Output:
                    {{"Topic": ["Food", "Restaurant"], "Location": ["Thành phố Hồ Chí Minh"]}}

                    Question: "Tôi muốn ăn phở Hà Nội tại Đà Nẵng"
                    Output:
                    {{"Topic": ["Food", "Restaurant"], "Location": ["Đà Nẵng"]}}
                    
                    ---

                    ### EXAMPLES (Demonstrating Comparisons and Standard Cases):
                    
                    Question: "Các khách sạn ở Đà Nẵng có gần biển không?"
                    Output:
                    {{"Topic": ["Accommodation"], "Location": ["Đà Nẵng"]}}

                    Question: "Ẩm thực Hà Nội và Đà Nẵng khác nhau thế nào?"
                    Output:
                    {{"Topic": ["Food"], "Location": ["Hà Nội", "Đà Nẵng"]}}

                    Question: "Khách sạn nào ở Thành phố Hồ Chí Minh gần chợ Bến Thành?"
                    Output:
                    {{"Topic": ["Accommodation"], "Location": ["Thành phố Hồ Chí Minh"]}}

                    Question: "Xin chào!"
                    Output:
                    {{"Topic": ["Off_topic"], "Location": []}}

                    ---

                Now classify the following Question:
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", "Question: {question}")
            ])

            classification_chain = prompt | llm | JsonOutputParser()

            print(f"\n---------------------Classifying query: {query}---------------------\n")
            classification = classification_chain.invoke({"question": query})

            # Filter allowed topics and locations
            allowed_cities = ["Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng"]
            allowed_topics = ['Food', 'Accommodation', 'Attraction', 'General', 'Festival', 'Restaurant', 'Transport', 'Plan', 'Off_topic']
            topics = [topic for topic in classification.get("Topic", []) if topic in allowed_topics]
            classification["Topic"] = topics
            locations = [loc for loc in classification.get("Location", []) if loc in allowed_cities]
            classification["Location"] = locations

            print(f"\n---------------------Classification result: {classification}---------------------\n")

            return classification

        except Exception as e:
            raise RuntimeError(f"Query classification error: {e}")

    @staticmethod
    def retrieve_documents(retriever, query: str, pinecone_reranker: PineconeRerank = None):
        try:
            start_time_retrieval = os.times()
            print("\n---------------------Retrieving relevant documents...---------------------\n")
            context_docs = retriever.invoke(query)
            end_time_retrieval = os.times()
            print("\n---------------------Retrieved relevant documents in", end_time_retrieval.user - start_time_retrieval.user, "seconds---------------------\n")

            # Pinecone Reranker
            if pinecone_reranker:
                print("\n---------------------Reranking documents with PineconeRerank...---------------------\n")
                start_time_reranking = os.times()
                context_docs = pinecone_reranker.compress_documents(documents=context_docs, query=query)
                end_time_reranking = os.times()
                print("\n---------------------Reranked documents in", end_time_reranking.user - start_time_reranking.user, "seconds---------------------\n")

            

            print("\n---------------------Context Documents:---------------------\n")
            for index, doc in enumerate(context_docs):
                if(doc.relevance_score is not None):
                    if doc.relevance_score < 0.3:
                        print(f"Removing low-relevance document (score: {doc.relevance_score}):\n {doc.page_content}\n")
                        context_docs.remove(doc)
                        continue
                if index > 0:
                    print("--------------------------------------------------------------\n")
                print(f"Context number {index}:\n {doc.page_content}")
                print("  Metadata:", doc.metadata)
            print("\n---------------------End of Context Documents---------------------\n")

            return context_docs

        except Exception as e:
            raise RuntimeError(f"Document retrieval error: {e}")

    @staticmethod
    def build_standalone_question(question: str, chat_history: list):
        try:
            contextualize_q_system_prompt = """Bạn là một công cụ viết lại câu. Nhiệm vụ duy nhất của bạn là chuyển đổi "Câu hỏi mới" và "Lịch sử trò chuyện" thành một "Câu hỏi độc lập" (standalone question) có thể hiểu được mà không cần lịch sử.

                Định dạng JSON BẮT BUỘC:
                ```json
                {{
                    "standalone_question": "Câu hỏi độc lập được viết lại ở đây"
                }}
                ```
            
                QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):

                1.  **NGHIÊM CẤM TRẢ LỜI CÂU HỎI:** Vai trò của bạn KHÔNG phải là trả lời. Nhiệm vụ chỉ là VIẾT LẠI CÂU HỎI vào trường JSON. Nếu bạn trả lời, bạn đã thất bại.
                2.  **NGHIÊM CẤM SAO CHÉP LỊCH SỬ:** KHÔNG được sao chép câu trả lời của AI từ "Lịch sử trò chuyện". Chỉ sử dụng "Lịch sử trò chuyện" để HIỂU NGHĨA và BỐI CẢNH của "Câu hỏi mới".
                3.  **QUY TẮC "VIẾT LẠI" (ƯU TIÊN SỐ 1):** Nếu "Câu hỏi mới" là câu hỏi ngắn, phụ thuộc vào lịch sử (ví dụ: "Ở đó giá bao nhiêu?", "Mấy giờ vậy?") HOẶC là một câu hỏi chung chung (ví dụ: "lên kế hoạch", "đi du lịch") mà bối cảnh địa điểm nằm trong Lịch sử, hãy dùng "Lịch sử trò chuyện" và bối cảnh địa điểm đó để viết lại thành câu hỏi đầy đủ và điền vào trường "standalone_question".
                4.  Nếu "Câu hỏi mới" chỉ gồm 1-2 từ ngắn gọn như “Tiếp”, “Còn gì?”, “Ở đó sao?”, “Món đó ngon không?”, hãy sử dụng “Lịch sử” để suy ra chủ đề hoặc địa điểm gần nhất và viết lại thành câu hỏi đầy đủ có ngữ nghĩa hoàn chỉnh.
                5.  **QUY TẮC "LẶP LẠI" (ƯU TIÊN SỐ 2):** Nếu "Câu hỏi mới" đã là một câu hỏi độc lập, đầy đủ ý nghĩa và không cần lịch sử, hãy điền Y HỆT nó vào trường "standalone_question".
                6.  **GIỚI HẠN OUTPUT:** CHỈ được xuất ra câu hỏi độc lập đã được viết lại. Không thêm lời chào, lời giải thích, hay bất cứ thứ gì khác.
                7.  **GIỮ NGUYÊN ĐẠI TỪ:** Phải giữ nguyên đại từ của người dùng ("tôi", "cho tôi", "của tôi").
                8.  **CHỌN NGỮ CẢNH MỚI NHẤT:** Nếu có nhiều câu trong "Lịch sử trò chuyện", chỉ sử dụng ngữ cảnh gần nhất của người dùng (Human) để viết lại câu hỏi. Không dựa vào các câu hỏi cũ hơn hoặc câu trả lời của AI nếu chúng không liên quan trực tiếp đến "Câu hỏi mới".
                9.  Nếu "Lịch sử" không chứa thông tin địa điểm, thời gian, hoặc chủ đề rõ ràng, giữ nguyên "Câu hỏi mới" mà không suy diễn thêm.

                BẮT BUỘC: Output phải là JSON hợp lệ duy nhất, không được chứa ký tự hoặc giải thích ngoài cặp dấu json.

                ---
                VÍ DỤ (Làm rõ QUY TẮC "LẶP LẠI"):
                ---
                Lịch sử: []
                Câu hỏi mới: "Hãy lên kế hoạch du lịch tại cần giờ"
                OUTPUT:
                {{
                    "standalone_question": "Hãy lên kế hoạch du lịch tại cần giờ"
                }}
                ---
                Lịch sử: []
                Câu hỏi mới: "Các món ăn ngon ở Hà Nội là gì?"
                OUTPUT:
                {{
                    "standalone_question": "Các món ăn ngon ở Hà Nội là gì?"
                }}
                ---
                Lịch sử: [Human: "Tôi muốn đi du lịch TPHCM"]
                Câu hỏi mới: "Hãy cho tôi danh sách các món ăn ngon tại hồ chí minh"
                OUTPUT:
                {{
                    "standalone_question": "Hãy cho tôi danh sách các món ăn ngon tại hồ chí minh"
                }}
                ---
                Lịch sử: []
                Câu hỏi mới: "Tại quận 4, con đường nào nổi tiếng với các quán hải sản nướng và món ốc đặc sản của TPHCM"
                OUTPUT:
                {{
                    "standalone_question": "Tại quận 4, con đường nào nổi tiếng với các quán hải sản nướng và món ốc đặc sản của TPHCM"
                }}
                ---

                VÍ DỤ (Làm rõ QUY TẮC "VIẾT LẠI"):
                ---
                Lịch sử: [Human: "cho tôi các món ăn nổi tiếng ở tphcm"\nAI: "TPHCM có món A, B, C..."]
                Câu hỏi mới: "Tôi muốn đi du lịch 3 ngày 2 đêm"
                OUTPUT:
                {{
                    "standalone_question": "Tôi muốn đi du lịch 3 ngày 2 đêm ở tphcm"
                }}
                ---
                Lịch sử: [Human: "Tôi muốn đi du lịch Thành phố Hồ Chí Minh"]
                Câu hỏi mới: "Ở đó có gì chơi?"
                OUTPUT:
                {{
                    "standalone_question": "Ở Thành phố Hồ Chí Minh có gì chơi?"
                }}
                ---
                Lịch sử: [Human: "Tôi muốn đi Hà Nội"]
                Câu hỏi mới: "lên kế hoạch du lịch 2 ngày"
                OUTPUT:
                {{
                    "standalone_question": "Lên kế hoạch du lịch 2 ngày ở Hà Nội"
                }}
                ---
                Lịch sử: [Human: "Cầu Rồng đẹp thật!"\nAI: "Đúng vậy, Cầu Rồng phun lửa vào cuối tuần."]
                Câu hỏi mới: "Mấy giờ vậy?"
                OUTPUT:
                {{
                    "standalone_question": "Cầu Rồng phun lửa vào mấy giờ?"
                }}
                ---

                Bây giờ, hãy thực hiện nhiệm vụ cho Lịch sử và Câu hỏi mới dưới đây:
            """

            history_lines = []
            for msg in chat_history:
                role = "Human" if msg.type == 'human' else "AI"
                history_lines.append(f"{role}: \"{msg.content}\"")
            
            chat_history_str = "\n".join(history_lines)
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
            contextualize_q_chain = contextualize_q_prompt | llm_create_standalone_question() | JsonOutputParser()

            standalone_question = contextualize_q_chain.invoke(
                {
                    "input": question,
                    "chat_history": chat_history_str
                }
            )

            return standalone_question
        except Exception as e:
            print(f"Error in building standalone question: {e}")
            return {"standalone_question": question}
        
    
    def classify_query_for_schedule(query: str):
        try:
            llm = llm_classify()

            # Prompt hệ thống đã được viết lại hoàn toàn cho nhiệm vụ mới
            system = """You are a specialized query classifier assistant.

                Your SOLE task is to analyze the user's "Question" to determine two things:
                1.  **Topic**: Is the question an explicit request for planning/scheduling?
                2.  **Location**: Is the location context one of the 3 allowed cities?

                ### CLASSIFICATION RULES:

                1.  **Topic**
                    -   You are ONLY allowed to identify one topic: "Plan".
                    -   A question is "Plan" IF AND ONLY IF it explicitly asks to create an **itinerary**, **schedule**, **tour**, or a **multi-day plan** (e.g., "2-day plan", "3 days 2 nights itinerary", "make a 1-day tour").
                    -   If it is a "Plan", output: `"Topic": "Plan"`
                    -   If the question just asks for options (e.g., "what to do in the morning?"), asks about dining, or anything else that is NOT an itinerary, it is **NOT** a "Plan". In this case, output: `"Topic": null`

                2.  **Location**
                    -   Represents only the main city context.
                    -   Allowed values: ['Hà Nội', 'Thành phố Hồ Chí Minh', 'Đà Nẵng'].
                    -   If one of these cities is **explicitly mentioned**, output that city name. Example: `"Location": "Hà Nội"`
                    -   If **no allowed city** is mentioned (or only a district/small area like "Cần Giờ", "Quận 5" is mentioned), you MUST output: `"Location": null`
                    -   Do NOT return a list. Only return a single string or `null`.

                3.  **Output Format**
                    -   Output ONLY valid JSON.
                    -   Structure:
                        ```jsonW
                        {{
                        "Topic": "Plan" | null,
                        "Location": "Hà Nội" | "Thành phố Hồ Chí Minh" | "Đà Nẵng" | null
                        }}
                        ```
                    -   No explanations.

                ---
                ### EXAMPLES

                Question: "Lên kế hoạch du lịch 2 ngày ở Hà Nội"
                Output:
                {{"Topic": "Plan", "Location": "Hà Nội"}}

                Question: "Tôi muốn đi Cần Giờ 1 ngày"
                Output:
                {{"Topic": "Plan", "Location": null}}

                Question: "Gợi ý lịch trình 3 ngày 2 đêm tại Đà Nẵng"
                Output:
                {{"Topic": "Plan", "Location": "Đà Nẵng"}}

                Question: "Những nơi vui chơi ở Sài Gòn buổi tối?"
                Output:
                {{"Topic": null, "Location": "Thành phố Hồ Chí Minh"}}

                Question: "Khách sạn nào gần chợ Bến Thành?"
                Output:
                {{"Topic": null, "Location": null}}

                Question: "Xin chào!"
                Output:
                {{"Topic": null, "Location": null}}

                Question: "Bánh mì Hà Nội được bán ở đâu ở Hồ Chí Minh?"
                Output:
                {{"Topic": null, "Location": "Thành phố Hồ Chí Minh"}}
                ---

                Now classify the following Question:
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", "Question: {question}")
            ])

            classification_chain = prompt | llm | JsonOutputParser()

            print(f"\n---------------------Classifying query for schedule: {query}---------------------\n")
            classification = classification_chain.invoke({"question": query})

            raw_topic = classification.get("Topic")
            raw_location = classification.get("Location")

            final_topic = None
            if raw_topic == "Plan":
                final_topic = "Plan"

            final_location = None
            allowed_cities = ["Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng"]
            if raw_location in allowed_cities:
                final_location = raw_location
                
            # Tạo kết quả cuối cùng, sạch sẽ
            final_result = {
                "Topic": final_topic,
                "Location": final_location
            }
            
            print(f"\n---------------------Classification result: {final_result}---------------------\n")

            return final_result

        except Exception as e:
            # Trả về None cho cả hai nếu có lỗi xảy ra
            print(f"Query classification error: {e}")
            return {"Topic": None, "Location": None}