import requests
import time
import csv
from datetime import datetime
from langchain.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator


# ====== Cấu hình ======
FILE_PATH = "./evaluate/hanoi.txt"   # file chứa danh sách câu hỏi
OUTPUT_CSV = "./evaluate/results_hanoi.csv"              # file CSV output
OUTPUT_MD = "./evaluate/results_hanoi.md"                # file Markdown output
MAX_LINES = 100                                     # số dòng tối đa cần đọc (None = đọc toàn bộ)
API_URL = "http://localhost:8080/ask"              # endpoint /ask

# ====== Hàm đọc câu hỏi ======
def read_questions_from_file(file_path: str, max_lines: int | None = None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # bỏ dòng trống
    if max_lines is not None:
        lines = lines[:max_lines]
    return lines

# ====== Hàm chạy đánh giá ======
def evaluate_rag():
    questions = read_questions_from_file(FILE_PATH, MAX_LINES)
    results = []

    print(f"🔍 Bắt đầu đánh giá {len(questions)} câu hỏi...\n")

    for idx, q in enumerate(questions, 1):
        try:
            start_time = time.time()
            response = requests.post(API_URL, json={"query": q})
            response.raise_for_status()
            elapsed = round(time.time() - start_time, 2)

            data = response.json()
            answer = data.get("answer", "Không có phản hồi")
            contexts = data.get("contexts", []) or data.get("context_docs", [])
            topk_text = " || ".join(contexts[:5]) if isinstance(contexts, list) else str(contexts)

            print(f"[{idx}] ❓ Câu hỏi: {q}")
            print(f"🧠 Trả lời: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"📚 Context (top k): {topk_text[:150]}{'...' if len(topk_text) > 150 else ''}")
            print(f"⏱️ Tốc độ phản hồi: {elapsed}s\n")

            results.append({
                "index": idx,
                "question": q,
                "answer": answer,
                "context(top_k)": topk_text,
                "response_time(s)": elapsed,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        except Exception as e:
            print(f"[{idx}] ❌ Lỗi với câu hỏi: {q}")
            print(f"Chi tiết lỗi: {e}\n")

            results.append({
                "index": idx,
                "question": q,
                "answer": "Lỗi API hoặc kết nối",
                "context(top_k)": "",
                "response_time(s)": "",
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    # ====== Ghi kết quả ra CSV ======
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ File CSV đã được tạo: {OUTPUT_CSV}")

    # ====== Ghi kết quả ra Markdown ======
    with open(OUTPUT_MD, "w", encoding="utf-8") as md:
        md.write(f"# 🧪 Kết quả đánh giá RAG ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
        for r in results:
            md.write(f"### {r['index']}. ❓ **Câu hỏi:** {r['question']}\n")
            md.write(f"- 🧠 **Trả lời:** {r['answer']}\n")
            md.write(f"- 📚 **Context (top-k):** {r['context(top_k)']}\n")
            md.write(f"- ⏱️ **Tốc độ phản hồi:** {r['response_time(s)']}s\n")
            md.write(f"- 🕒 {r['datetime']}\n\n")
            md.write("---\n\n")
    print(f"✅ File Markdown đã được tạo: {OUTPUT_MD}")

    print("\n🎯 Hoàn tất! Kết quả gồm cả CSV và Markdown đã sẵn sàng.\n")

if __name__ == "__main__":
    evaluate_rag()
