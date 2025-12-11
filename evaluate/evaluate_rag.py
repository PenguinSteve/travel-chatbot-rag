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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

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

# Hàm tạo answer và trích xuất context RAG cho từng câu hỏi trong tập đánh giá
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

### Hàm hỗ trợ đánh giá BERTScore
# Hàm làm sạch định dạng Markdown trong văn bản
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
    text = text.replace('\n', '. ')
    
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

### END OF BERTScore functions ###

### CÁC HÀM HỖ TRỢ ĐÁNH GIÁ GROUNDEDNESS
def split_sentences(text):
    """Tách đoạn văn thành các câu đơn để kiểm tra từng ý."""
    if not isinstance(text, str): return []
    # Tách dựa trên dấu câu kết thúc (. ? !) và theo sau là khoảng trắng
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

### Hàm tạo sliding windows cho context dài
def create_sliding_windows(text, window_size=350, overlap=50):
    """
    Chia Context dài thành các cửa sổ trượt (Sliding Windows) để xử lý Parent Document.
    Đơn vị: Số từ (words).
    """
    if not isinstance(text, str): return [""]
    words = text.split()
    if len(words) <= window_size:
        return [text]
    
    windows = []
    for i in range(0, len(words), window_size - overlap):
        chunk = " ".join(words[i : i + window_size])
        windows.append(chunk)
        if i + window_size >= len(words):
            break
    return windows

### Hàm đánh giá Groundedness sử dụng NLI Cross-Encoder
def calculate_Groundedness(input_path: str, output_path: str):
    """
    Tính điểm Groundedness sử dụng mô hình NLI (Cross-Encoder) và Sliding Window.
    """
    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ Groundedness ---")
    
    # 1. Load Model NLI
    model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    print(f"Đang tải model NLI: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị chạy: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return

    # 2. Load Dữ liệu
    print(f"Đọc dữ liệu từ: {input_path}")
    df = pd.read_excel(input_path).fillna("")

    # df = df[0:10]  # Giới hạn xử lý 10 dòng đầu cho test

    # Kiểm tra cột Context. Trong code của bạn, cột này tên là 'Context_Documents'
    context_col = 'Context_Documents'
    if context_col not in df.columns:
        # Fallback: Tìm các cột có chữ Context
        cols = [c for c in df.columns if 'Context' in c]
        if cols:
            context_col = cols[0]
            print(f"Không thấy 'Context_Documents', dùng cột thay thế: {context_col}")
        else:
            print("Lỗi: Không tìm thấy cột chứa Context.")
            return

    scores = []
    details_list = []

    SKIP_PATTERNS = [
        # --- NHÓM 1: THÔNG TIN TIỆN ÍCH (Hybrid Knowledge) ---
        # Giá tiền: Bắt các cụm "50.000 VND", "20k", "Miễn phí"
        r'(?:giá|mức giá|chi phí).*?(?:tham khảo|vé|dịch vụ)?[:\s]*[\d.,]+.*?(?:vnđ|vnd|đ|k|usd)', 
        r'\d{1,3}(?:[.,]\d{3})*\s*(?:vnđ|vnd|đ|k)', # Bắt số tiền đứng lẻ (50.000 VND)
        r'(?:giá|vé).*?(?:miễn phí|tự do)',         # Bắt chữ "miễn phí"
        
        # Thời gian: Bắt giờ mở cửa (07:00 – 22:00)
        r'(?:giờ|thời gian).*?(?:mở cửa|hoạt động)[:\s]*\d{1,2}[:h]',
        r'\d{1,2}[:h]\d{2}\s*[–-]\s*\d{1,2}[:h]\d{2}', # Format 07:00 – 22:00

        # Liên hệ & Địa chỉ (Thường Context có địa chỉ nhưng AI hay viết lại khác format -> Skip cho an toàn)
        r'(?:liên hệ|sđt|hotline|điện thoại)[:\s]*0\d+',
        r'địa chỉ[:\s].*', 

        # --- NHÓM 2: VĂN PHONG SÁO RỖNG (Fluff / Hallucination cảm xúc) ---
        # Câu kết bài xã giao
        r'(?:chúc|hy vọng|mong).*?(?:bạn|du khách).*?(?:chuyến|trải nghiệm|vui vẻ|thú vị)',
        r'(?:hãy|đừng quên).*?(?:đến|ghé|thử|mang theo)',
        r'cảm ơn bạn đã quan tâm',
        
        # Các câu mô tả cảm giác chủ quan (thường gây lỗi NLI)
        r'(?:mang lại|tạo nên|đem đến).*?(?:cảm giác|không gian|trải nghiệm).*?(?:tuyệt vời|lý tưởng|hoàn hảo|trọn vẹn)',
        r'là (?:điểm đến|lựa chọn) (?:lý tưởng|hoàn hảo|tuyệt vời)',
        r'thu hút (?:đông đảo)? du khách',
        r'nổi tiếng với vẻ đẹp',
        
        # Câu dẫn dắt vô nghĩa
        r'dưới đây là',
        r'thông tin chi tiết',
        r'một số gợi ý'
    ]

    def calculate_overlap(sent, context_text):
        # Tách từ đơn giản, bỏ qua các từ quá ngắn (<2 ký tự)
        sent_tokens = set([w.lower() for w in sent.split() if len(w) > 1])
        ctx_tokens = set([w.lower() for w in context_text.split() if len(w) > 1])
        
        if len(sent_tokens) == 0: return 0.0
        
        intersection = sent_tokens.intersection(ctx_tokens)
        return len(intersection) / len(sent_tokens)

    print("Đang chấm điểm từng câu trả lời...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        answer = str(row['Answer'])
        # Làm sạch context một chút trước khi xử lý (bỏ tên tài liệu, score nếu cần thiết)
        full_context = str(row[context_col])
        
        # A. Tách câu trả lời thành các mệnh đề/câu đơn
        overlap_context_ref = full_context.lower()

        answer = clean_markdown(answer)
        sentences = split_sentences(answer)
        if not sentences:
            scores.append(0.0)
            details_list.append("No valid sentences")
            continue

        # B. Tạo cửa sổ trượt cho Context (Xử lý Context siêu dài từ Parent Document)
        # Window size 300 từ ~ 450-500 tokens (an toàn cho mDeBERTa 512)
        context_windows = create_sliding_windows(full_context, window_size=300, overlap=50)
        
        pass_count = 0
        skipped_count = 0
        row_details = []

        # C. So khớp từng câu Answer với các cửa sổ Context
        for sent in sentences:

            # Filter các câu không cần đánh giá (Thông tin tiện ích, văn phong rỗng)
            is_external_info = False
            
            # Kiểm tra câu có quá ngắn không (dưới 3 từ thường là rác do tách câu sai)
            if len(sent.split()) < 3: 
                is_external_info = True
            
            # Kiểm tra Regex
            if not is_external_info:
                for pattern in SKIP_PATTERNS:
                    if re.search(pattern, sent, re.IGNORECASE):
                        is_external_info = True
                        break
            
            if is_external_info:
                skipped_count += 1
                row_details.append(f"SKIP: {sent[:40]}...")
                continue

            # Đánh giá câu này với tất cả cửa sổ Context
            max_entailment = -1.0
            
            # Quét qua các cửa sổ để tìm bằng chứng tốt nhất (Max Score Strategy)
            for window in context_windows:
                # Input format cho Cross-Encoder: [CLS] Context [SEP] Sentence [SEP]
                inputs = tokenizer(
                    window, sent, 
                    truncation=True, return_tensors="pt", max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = model(**inputs)
                    probs = torch.softmax(output.logits, dim=1).tolist()[0]
                
                # Model MoritzLaurer output: 0: Entailment, 1: Neutral, 2: Contradiction
                entailment_score = probs[0]
                
                if entailment_score > max_entailment:
                    max_entailment = entailment_score
                
                # Tối ưu: Nếu tìm thấy bằng chứng chắc chắn (>0.9), dừng tìm kiếm cho câu này
                if max_entailment > 0.9: 
                    break
            
            # Tính điểm Overlap từ vựng giữa câu và toàn bộ Context
            overlap_score = calculate_overlap(sent, overlap_context_ref)

            is_pass = False
            note = ""

            # Đánh giá câu này
            # Ngưỡng 0.35 là ngưỡng phù hợp để xác định câu bị paraphrase
            if max_entailment > 0.35: 
                is_pass = True
                note = f"PASS(NLI={max_entailment:.2f}), sent: {sent}"
            elif overlap_score > 0.5:
                is_pass = True
                note = f"PASS(Overlap={overlap_score:.2f}), sent: {sent}"
            else:
                is_pass = False
                note = f"FAIL(NLI={max_entailment:.2f}, Ov={overlap_score:.2f})"

            if is_pass:
                    pass_count += 1
                    row_details.append(f"{note}")
            else:
                row_details.append(f"{note}: {sent}...")

        effective_total = len(sentences) - skipped_count
        
        if effective_total == 0:
            final_row_score = 1.0
            row_details.append("(All sentences skipped)")
        else:
            final_row_score = pass_count / effective_total

        # D. Tính điểm trung bình cho dòng này (Tỷ lệ câu đúng)
        scores.append(final_row_score)
        details_list.append("; ".join(row_details))

    # 3. Lưu kết quả
    df['Groundedness_Score'] = scores
    df['Groundedness_Details'] = details_list
    
    df.to_excel(output_path, index=False)
    
    avg_score = sum(scores)/len(scores) if scores else 0
    print(f"\n--- HOÀN TẤT GROUNDEDNESS ---")
    print(f"Điểm Groundedness trung bình: {avg_score:.4f}")
    print(f"Kết quả lưu tại: {output_path}")

### MAIN FUNCTION ###
# --- --------------- Example usage --------------- ---
if __name__ == "__main__":
    
    input_path = "./evaluate/data/data_evaluate_TPHCM.xlsx"

    # evaluate(input_path, output_path="rag_evaluation_results_TPHCM_1.xlsx")

    # calculate_BERTScore(input_path="./evaluate/result/rag_evaluation_results_DaNang.xlsx",
    #                     output_path="model_microsoft-mdeberta-v3-base_layer11_rag_evaluation_results_DaNang_with_BERTScore.xlsx")

    calculate_Groundedness(
        input_path="./evaluate/result/rag_evaluation_results_Hanoi.xlsx",
        output_path="rag_evaluation_results_Hanoi_with_Groundedness_test_1.xlsx"
    )