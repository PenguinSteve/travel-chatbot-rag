import pandas as pd
from typing import List, Dict, Any
from app.config.settings import settings
from app.config.vector_database_pinecone import PineconeConfig
from app.services.rag_service import RAGService
from app.core.llm import llm_rag
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.storage import MongoDBStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeRerank
from bert_score import score
import re

class RAGEvaluation:
    CONNECTION_STRING = f"mongodb+srv://{settings.MONGO_DB_NAME}:{settings.MONGO_DB_PASSWORD}@chat-box-tourism.ojhdj0o.mongodb.net/?retryWrites=true&w=majority&tls=true"

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    def __init__(self):
        vector_store = PineconeConfig().get_vector_store()
        self.docstore = MongoDBStore(
            connection_string=self.CONNECTION_STRING,
            db_name=settings.MONGO_DB_NAME,
            collection_name=settings.MONGO_STORE_COLLECTION_NAME
        )
        self.parent_document_retriever = ParentDocumentRetriever(
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            vectorstore=vector_store,
            search_kwargs={"k":15, "filter":{}}
        )
        self.pinecone_reranker = PineconeRerank(top_n=3, pinecone_api_key=settings.PINECONE_API_KEY_RERANKER)

    # Implement RAG response generation for evaluation
    def generate_response(self, retriever, question: str, topics, locations) -> str:
        llm = llm_rag()

        system = """### Role and Goal
                You are an AI assistant specializing in tourism. Your persona is friendly, helpful, **detailed** and **extremely accurate**.
                Your task is to answer user questions using a Hybrid approach: Prioritize the provided 'Context', but supplement with your own knowledge when specific details are missing.
                Your goal is to provide the **most comprehensive answer possible**.

                ### The Golden Rules (MOST IMPORTANT)

                1.  **Source Hierarchy (Context First, Knowledge Second):**
                    * **Core Entities:** The list of places, restaurants, dishes, or activities you recommend MUST come **EXCLUSIVELY** from the 'Context'. **DO NOT** invent new places or recommend items not mentioned in the 'Context'.
                    * **Supplementary Details [IMPORTANT EXCEPTION]:** If a place/item is found in the 'Context' but specific factual details (specifically: **Address**, **Price**, **Opening Hours**, or **Contact Info**) are missing, you **ARE AUTHORIZED AND ENCOURAGED** to use your internal pre-trained knowledge to fill in these missing details.
                    * **[CRITICAL - SEAMLESS BLENDING]:** You must blend these two sources (Context & Internal Knowledge) into a single, unified voice. **DO NOT** distinguish between them in your output. The user must NOT know which part came from the document and which came from your internal knowledge.
                    * If you use internal knowledge for Price/Hours, ensure it is the most recent estimate you know.

                2.  **Natural Phrasing (No Meta-Talk):**
                    * You must sound like a natural, human expert.
                    * **DO NOT** talk about yourself as an AI, mention "Context," "documents," or "internal knowledge."
                    * **STRICTLY FORBIDDEN PHRASES:** You are prohibited from using phrases like:
                        * "Thông tin này được bổ sung từ kiến thức chung" (This info added from general knowledge)
                        * "Trong ngữ cảnh không có thông tin này" (Context lacks this info)
                        * "Lưu ý: Dữ liệu về giá được lấy từ..." (Note: Price data is taken from...)
                    * Just state the information directly as if you know it.

                3.  **The "Be Detailed and Helpful" Rule:**
                    * When the user asks for information, find relevant items in the 'Context'.
                    * **[QUANTITY LOGIC]:**
                        * **Case A (General Request):** If the user asks a general question **WITHOUT** specifying a quantity (e.g., "suggest some places"), **you MUST limit your response to the top 3-5 most relevant items** found in the 'Context' to avoid overwhelming the user.
                        * **Case B (Specific Request):** If the user specifies a quantity (e.g., "top 10", "list all"), follow that instruction.
                        * **Case C (Insufficient Data):** If the 'Context' has fewer items than requested (e.g., user asks for 20, Context has 10), detailedly describe the 10 items you have and naturally state that those are the best recommendations you have right now.
                    * **[DETAIL REQUIREMENT]:** For the items you choose to list, provide a detailed summary:
                        * *Step 1:* Extract description/facts from 'Context'.
                        * *Step 2:* Check if 'Address' or 'Price' is missing in 'Context'.
                        * *Step 3:* If missing, retrieve them from your internal knowledge to make the answer complete.

                4.  **[PRIORITY 1] Handling Off-topic, Greeting, or Vague Questions:**
                    * Check this **FIRST**.
                    * If Greeting/Off-topic: Respond politely and steer back to tourism.
                    * If Vague (e.g., "Give me info"): Respond with: "Tôi có thể giúp gì cho bạn về thông tin du lịch?".

                5.  **Handling Conversation History:**
                    * Use 'Conversation History' to understand follow-ups.
                    * Avoid repeating information unless asked.

                6.  **Handling **Completely** Missing Topics:**
                    * This rule applies ONLY if the 'Context' contains **NO relevant places/items** related to the user's question.
                    * In this specific case (Context is empty regarding the topic), respond with: "Tôi hiện chưa thể đưa ra câu trả lời chính xác vì dữ liệu liên quan đến yêu cầu của bạn chưa đầy đủ. Bạn hãy cung cấp thêm thông tin (như nguồn dữ liệu, nội dung cụ thể hoặc ví dụ minh hoạ) để tôi có thể hỗ trợ bạn hiệu quả hơn."

                7.  **Formatting & Language:**
                    * **Language:** Always answer in **Vietnamese**.
                    * **No Post-amble:** Do not add any summary sentences at the end explaining where the information came from. **ABSOLUTELY NO** "Lưu ý" (Note) section about data sources.
                    * **Formatting:** Use clean **Markdown**.
                        * Use **Bold** (`**text**`) for names/highlights.
                        * Use **Bullet points** (`*` or `-`) for lists.
                        * **STRICTLY PROHIBITED:** No HTML tags (`<br>`, `<div>`). Remove or replace them if found in Context.
                """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", """'Ngữ cảnh' để bạn tham khảo.
                <Ngữ cảnh>
                {context}
                </Ngữ cảnh>

                Hãy trả lời 'Câu hỏi' dưới đây dựa trên các quy tắc đã đề ra.
                Câu hỏi: {question}""")
        ])

        rag_chain = prompt | llm | StrOutputParser()

        if( 'Off_topic' in topics ):
            prompt_input = {
                "context": "",
                "question": question
            }
            
            response = rag_chain.invoke(prompt_input)

            return response

        context_docs = RAGService.retrieve_documents(retriever, question, self.pinecone_reranker)

        formatted_contexts = []
        for doc in context_docs:
            name = doc.metadata.get('Name', 'Không rõ')
            
            context_str = f"Tên tài liệu: {name}\nNội dung: {doc.page_content}"
            formatted_contexts.append(context_str)

        prompt_input = {
            "context": "\n\n".join(formatted_contexts),
            "question": question
        }

        response = rag_chain.invoke(prompt_input)

        return response, context_docs

    def BERTScore(cands: List[str], refs: List[str], num_layers: int = 10, model_type: str = "microsoft/mdeberta-v3-base") -> List[float]:

        P, R, F1 = score(cands,
                        refs,
                        model_type=model_type,
                        lang="vi",
                        num_layers=num_layers,
                        batch_size=4,
                        verbose=True,
                        rescale_with_baseline=True)
        return P.tolist(), R.tolist(), F1.tolist()

def evaluate(input_path: str, output_path: str = "rag_evaluation_results_DaNang.xlsx"):
    print("Loading evaluation dataset from:", input_path)
    df = pd.read_excel(input_path).fillna("")

    df = df[99:]  # Process only one row for testing
    
    print(df.iloc[0])
    print(df.head())

    print(f"Evaluation dataset loaded. Total questions: {len(df)}")

    print("Initializing RAG evaluation components...")
    RAGEvaluation_instance = RAGEvaluation()

    results = []

    
    try:
        for index, row in df.iterrows():
            question = str(row.get("Question", ""))
            groundTruth = str(row.get("GroundTruth", ""))

            classify_result = RAGService.classify_query(question)

            topics = classify_result.get("Topic") or []
            locations = classify_result.get("Location") or []

            filter = {}
            if isinstance(topics, list) and len(topics) > 0:
                filter["Topic"] = {"$in": topics}
            if isinstance(locations, list) and len(locations) > 0:
                filter["Location"] = {"$in": locations}

            # Initialize retriever with filter
            retriever = RAGEvaluation_instance.parent_document_retriever
            retriever.search_kwargs["filter"] = filter

            # Generate response
            answer, context_docs = RAGEvaluation_instance.generate_response(
                retriever, question, topics, locations
            )

            # Prepare context string
            formatted_contexts = []
            for doc in context_docs:
                name = doc.metadata.get('Name', 'Không rõ')

                context_str = f"Tên tài liệu: {name}\nNội dung: {doc.page_content}\nScore: {doc.metadata.get('relevance_score', 'N/A')}"
                formatted_contexts.append(context_str)

            context_combined = "\n\n".join(formatted_contexts)

            # Store results
            results.append({
                "Question": question,
                "GroundTruth": groundTruth,
                "Answer": answer,
                "Context_Documents": context_combined
            })
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Saving partial results...")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\nEvaluation finished. Results saved to {output_path}")
    return df

def clean_markdown(text: str) -> str:
    if not isinstance(text, str): return str(text)
    
    # Loại bỏ hàng phân cách bảng (Ví dụ: |---|---| hoặc | :--- | :--- |)
    # Regex này tìm các dòng chỉ chứa dấu gạch đứng, gạch ngang, dấu hai chấm và khoảng trắng
    text = re.sub(r'\|[\s\-:|]+\|', ' ', text)
    
    # Loại bỏ ký tự tạo cột '|' -> thay bằng khoảng trắng để các từ không dính vào nhau
    # Ví dụ: "|Khách sạn|" -> " Khách sạn "
    text = text.replace('|', ' ')
    
    # Loại bỏ in đậm/nghiêng (Logic cũ)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  
    text = re.sub(r'\*(.*?)\*', r'\1', text)      
    text = re.sub(r'__(.*?)__', r'\1', text)      
    
    # Loại bỏ gạch đầu dòng (Logic cũ - Cần chạy khi còn Newline)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ tiêu đề (Logic cũ)
    text = re.sub(r'#+\s+', '', text)
    
    # Loại bỏ xuống dòng bằng khoảng trắng (Logic cũ)
    text = text.replace('\n', ' ')
    
    # Chuẩn hóa dấu câu (Logic cũ - Rất quan trọng để tiết kiệm token)
    
    # Xóa khoảng trắng trước dấu câu: " ." -> ".", " ," -> ","
    text = re.sub(r'\s+([.,;?!])', r'\1', text)
    
    # Gộp nhiều dấu chấm thành 1: ".." -> ".", ". ." -> "."
    text = re.sub(r'\.{2,}', '.', text)      # .. -> .
    text = re.sub(r'\.\s+\.', '.', text)     # . . -> .
    
    # Sửa lỗi dấu hai chấm kèm chấm: ":." -> ":"
    text = re.sub(r':\.', ':', text)
    
    # Loại bỏ khoảng trắng thừa (Logic cũ)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example Cosine Similarity Matrix
def visualize_matrix(candidate, reference, num_layers=10, model_type="microsoft/mdeberta-v3-base"):
    from bert_score import plot_example
    import matplotlib.pyplot as plt

    if len(candidate) > 200:
        candidate = candidate[:30] + "..."
    if len(reference) > 200:
        reference = reference[:30] + "..."

    plot_example(candidate,
                reference, 
                model_type=model_type, 
                lang="vi",
                num_layers=num_layers)
    
    plt.title(f"BERTScore Cosine Similarity Matrix with layer {num_layers}")
    plt.show()

    plot_example(candidate,
                reference, 
                model_type=model_type, 
                lang="vi",
                num_layers=num_layers+1)
    plt.title(f"BERTScore Cosine Similarity Matrix with layer {num_layers+1}")
    plt.show()

    plot_example(candidate,
                reference, 
                model_type=model_type, 
                lang="vi",
                num_layers=num_layers+2)
    plt.title("BERTScore Cosine Similarity Matrix with layer 12")
    plt.show()

def truncated_text(text, max_length=300):
    words = text.split()

    if len(words) <= max_length:
        return text
    
    truncated = ' '.join(words[:max_length]) + '...'
    return truncated
    

def calculate_BERTScore(input_path: str, output_path: str = "rag_evaluation_results_with_BERTScore.xlsx"):
    print("Loading evaluation results from:", input_path)
    df = pd.read_excel(input_path).fillna("")

    # df = df[0:1] 

    print(f"Evaluation results loaded. Total entries: {len(df)}")

    if 'Answer' not in df.columns or 'GroundTruth' not in df.columns:
        print("Lỗi: File Excel thiếu cột 'Answer' hoặc 'GroundTruth'.")
        return

    cands = df["Answer"].astype(str).tolist()
    refs = df["GroundTruth"].astype(str).tolist()

    cands = [clean_markdown(text) for text in cands]
    refs = [clean_markdown(text) for text in refs]

    # visualize_matrix(cands, refs, num_layers=10, model_type="uitnlp/visobert")

    print("Calculating BERTScore...")
    P, R, F1 = RAGEvaluation.BERTScore(cands, refs, num_layers=11, model_type="microsoft/mdeberta-v3-base")

    df["BERTScore_Precision"] = P
    df["BERTScore_Recall"] = R
    df["BERTScore_F1"] = F1

    df.to_excel(output_path, index=False)
    print(f"\nBERTScore calculation finished. Results saved to {output_path}")
    return df


# --- --------------- Example usage --------------- ---
if __name__ == "__main__":
    
    input_path = "./evaluate/data/data_evaluate_TPHCM.xlsx"

    # evaluate(input_path, output_path="rag_evaluation_results_TPHCM_1.xlsx")

    calculate_BERTScore(input_path="./evaluate/result/rag_evaluation_results_DaNang.xlsx",
                        output_path="model_microsoft-mdeberta-v3-base_layer11_rag_evaluation_results_DaNang_with_BERTScore.xlsx")