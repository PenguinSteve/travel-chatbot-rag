from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

class RerankerService:
    model_name: str

    def __init__(self, model_name: str = 'AITeamVN/Vietnamese_Reranker', device = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = 2304
        
        print(f"\n---------------------Reranker model '{model_name}' loaded on {self.device}---------------------\n")

    def rerank(self, query: str, docs: list[str], top_k: int = 5):
        if not docs:
            return []
        
        pairs = [ (query, doc.page_content) for doc in docs ]

        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)

        for doc, score in ranked:
            doc.metadata['rerank_score'] = score
            
        top_docs = [ doc for doc, score in ranked]

        return top_docs[:top_k]