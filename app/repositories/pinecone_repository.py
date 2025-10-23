from app.config.settings import settings

class PineconeRepository:
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.default_k = settings.DEFAULT_RAG_TOP_K

        print(f"\n---------------------PineconeRepository initialized with default_k={self.default_k}---------------------\n")

    def get_retriever(self, k: int = None):
        k = k or self.default_k
        print(f"\n---------------------Creating retriever with top {k} documents---------------------\n")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        print(f"\n---------------------Retriever created with top {k} documents---------------------\n")
        return retriever