import os
import pandas as pd
from typing import List, Dict, Any
from app.config.settings import settings
from app.config.vector_database_pinecone import PineconeConfig
from app.services.rag_service import RAGService
from app.core.llm import llm_rag, llm_evaluate_faithfulness, llm_evaluate_relevance, llm_evaluate_precision, llm_evaluate_recall
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.storage import MongoDBStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        self.flashrank_comp = FlashrankRerank(top_n=3, model="ms-marco-MultiBERT-L-12")
        self.llm_rag_eval_faithfulness = llm_evaluate_faithfulness()
        self.llm_rag_eval_relevance = llm_evaluate_relevance()
        self.llm_rag_eval_precision = llm_evaluate_precision()
        self.llm_rag_eval_recall = llm_evaluate_recall()
        self.parent_document_retriever = ParentDocumentRetriever(
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            vectorstore=vector_store,
            search_kwargs={"k":5, "filter":{}}
        )

    # Implement RAG response generation for evaluation
    def generate_response(self, retriever, question: str, topics, locations) -> str:
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
            6.  **Handling Off-topic/Greeting:** If the 'Question' is a greeting or unrelated to tourism, respond politely, be friendly, and steer the conversation back to tourism (e.g., "Hello, how can I help you with your travel plans today?").
            7. No Post-amble: Do not add any summary sentences at the end explaining where the information came from. Just provide the direct answer.
            8.  **Language:** You must always answer in Vietnamese.
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
        
        context_docs = RAGService.retrieve_documents(retriever, question)

        formatted_contexts = []
        for doc in context_docs:
            name = doc.metadata.get('Name', 'Không rõ')
            
            context_str = f"Tên tài liệu: {name}\nNội dung: {doc.page_content}"
            formatted_contexts.append(context_str)

        # prompt_input = {
        #     "context": "\n\n".join(formatted_contexts),
        #     "question": question
        # }

        # response = rag_chain.invoke(prompt_input)
        response = ""

        return response, context_docs
    
    # --------------- Faithfulness Evaluation --------------- #
    # đánh giá mức độ trung thực của câu trả lời dựa trên ngữ cảnh được cung cấp
    def evaluate_faithfulness(self, question: str, answer: str, context: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator assessing factual accuracy between a model answer and its context."),
            ("user", """Evaluate the FAITHFULNESS of the following answer.

                Context:
                {context}

                Answer:
                {answer}

                Question:
                {question}

                Score between 0.0 (completely unfaithful, fabricated) and 1.0 (fully faithful, strictly supported by context).
                Explain briefly why.

                Output STRICTLY in JSON:
                {{"score": float, "explanation": "text"}}
            """)
        ])

        evaluate_faithfulness_chain = (prompt | self.llm_rag_eval_faithfulness | JsonOutputParser())
        return evaluate_faithfulness_chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })

    # --------------- Answer Relevance Evaluation --------------- #
    # đo lường mức độ liên quan của câu trả lời với câu hỏi
    def evaluate_answer_relevance(self, question: str, answer: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise evaluator of relevance between question and answer."),
            ("user", """Evaluate ANSWER RELEVANCE for the given pair:

                Question:
                {question}

                Answer:
                {answer}

                Score between 0.0 (irrelevant) and 1.0 (fully relevant and focused).
                Briefly justify.

                Output STRICTLY in JSON:
                {{"score": float, "explanation": "text"}}
            """)
        ])

        evaluate_relevance_chain = (prompt | self.llm_rag_eval_relevance | JsonOutputParser())
        return evaluate_relevance_chain.invoke({
            "question": question,
            "answer": answer
        })
    

    # --------------- Context Precision Evaluation --------------- #
    # đo lường mức độ chính xác của các đoạn ngữ cảnh (contexts) được truy xuất
    def evaluate_context_precision(self, question: str, context: str, answer: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are evaluating context precision — whether the retrieved context precisely supports the answer."),
            ("user", """Evaluate CONTEXT PRECISION.

                Question:
                {question}

                Answer:
                {answer}

                Context:
                {context}

                Score between 0.0 (context mostly irrelevant or noisy) and 1.0 (context concise and directly relevant).
                Explain in short.

                Output STRICTLY in JSON:
                {{"score": float, "explanation": "text"}}
            """)
        ])

        evaluate_precision_chain = (prompt | self.llm_rag_eval_precision | JsonOutputParser())
        return evaluate_precision_chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
    
    # --------------- Context Recall Evaluation --------------- #
    # đo lường mức độ đầy đủ của các đoạn ngữ cảnh (contexts) được truy xuất
    def evaluate_context_recall(self, question: str, context: str, ground_truth: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are evaluating how much of the ground truth information is present in the context."),
            ("user", """Evaluate CONTEXT RECALL.

                Ground Truth:
                {ground_truth}

                Question:
                {question}

                Context:
                {context}

                Score between 0.0 (context misses most key facts) and 1.0 (context covers all key facts needed).
                Give short justification.

                Output STRICTLY in JSON:
                {{"score": float, "explanation": "text"}}
            """)
        ])

        evaluate_recall_chain = (prompt | self.llm_rag_eval_recall | JsonOutputParser())
        return evaluate_recall_chain.invoke({
            "context": context,
            "question": question,
            "ground_truth": ground_truth
        })


def evaluate(input_path: str, output_path: str = "rag_evaluation_results_DaNang.xlsx"):
    print("Loading evaluation dataset from:", input_path)
    df = pd.read_excel(input_path).fillna("")

    # df = df[30:]  # Chạy thử từ dòng 102 đến hết
    
    # print(df)

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
            if 'Plan' not in topics:
                answer, context_docs = RAGEvaluation_instance.generate_response(retriever, question, topics, locations)
            else:
                answer = "N/A for planning questions in this evaluation."
                context_docs = []

            # Prepare context string
            formatted_contexts = []
            for doc in context_docs:
                name = doc.metadata.get('Name', 'Không rõ')
                
                context_str = f"Tên tài liệu: {name}\nNội dung: {doc.page_content}"
                formatted_contexts.append(context_str)

            context_combined = "\n\n".join(formatted_contexts)

            print(f"Context combined for evaluation:\n{context_combined}")

            # # Evaluate metrics faithfulness
            # faithfulness_result = RAGEvaluation_instance.evaluate_faithfulness(
            #     question, answer, context_combined
            # )
            # print(f"\nEvaluated row {index + 1}: Faithfulness score = {faithfulness_result.get('score') or 'N/A'}")
            # print(f"Faithfulness explanation: {faithfulness_result.get('explanation') or ''}")

            # # Evaluate metrics relevance
            # relevance_result = RAGEvaluation_instance.evaluate_answer_relevance(
            #     question, answer
            # )
            # print(f"\nEvaluated row {index + 1}: Relevance score = {relevance_result.get('score') or 'N/A'}")
            # print(f"Relevance explanation: {relevance_result.get('explanation') or ''}")

            # # Evaluate metrics context precision and recall
            # precision_result = RAGEvaluation_instance.evaluate_context_precision(
            #     question, context_combined, answer
            # )
            # print(f"\nEvaluated row {index + 1}: Context Precision score = {precision_result.get('score') or 'N/A'}")
            # print(f"Context Precision explanation: {precision_result.get('explanation') or ''}")

            # # Evaluate context recall
            # recall_result = RAGEvaluation_instance.evaluate_context_recall(
            #     question, context_combined, groundTruth
            # )
            # print(f"\nEvaluated row {index + 1}: Context Recall score = {recall_result.get('score') or 'N/A'}")
            # print(f"Context Recall explanation: {recall_result.get('explanation') or ''}")

            # # Store results
            # results.append({
            #     "Question": question,
            #     "GroundTruth": groundTruth,
            #     "Answer": answer,
            #     "Faithfulness_Score": faithfulness_result.get("score") or "N/A",
            #     "Faithfulness_Explanation": faithfulness_result.get("explanation") or "",
            #     "Relevance_Score": relevance_result.get("score") or "N/A",
            #     "Relevance_Explanation": relevance_result.get("explanation") or "",
            #     "Context_Precision_Score": precision_result.get("score") or "N/A",
            #     "Context_Precision_Explanation": precision_result.get("explanation") or "",
            #     "Context_Recall_Score": recall_result.get("score") or "N/A",
            #     "Context_Recall_Explanation": recall_result.get("explanation") or "",
            #     "Context_Documents": context_combined
            # })
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Saving partial results...")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\nEvaluation finished. Results saved to {output_path}")
    return df

# --- --------------- Example usage --------------- ---
if __name__ == "__main__":
    
    input_path = "./evaluate/data/data_evaluate_TPHCM.xlsx"

    evaluate(input_path, output_path="rag_evaluation_results_TPHCM.xlsx")
