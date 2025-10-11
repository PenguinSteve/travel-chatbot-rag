from langchain_groq import ChatGroq
from storedata import connect_to_pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_retriever(index_name: str, k: int = 5):
    vector_store = connect_to_pinecone(index_name=index_name)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

def generate_groq_response(retriever, query: str):
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    llm = ChatGroq(model=llm_model, temperature=0, api_key=groq_api_key)

    system = """You are an AI assistant that helps people find information about tourism.
    You are given the following extracted parts of a long document and a question.
    Provide a conversational answer based on the context provided.
    If you don't know the answer, just say "I don't know". Don't try to make up an answer.
    Always answer in Vietnamese.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])

    context_docs = retriever.invoke(query)
    prompt_input = {
        "context": "\n\n".join([doc.page_content for doc in context_docs]),
        "question": query
    }

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(prompt_input)
    return response, context_docs

def main():
    index_name = "rag-tourism"
    retriever = get_retriever(index_name=index_name, k=5)

    query = "Hà Nội có những địa điểm du lịch nổi tiếng nào?"
    response, context_docs = generate_groq_response(retriever, query)
    print("Query:", query)
    print("\n")
    print("---"*20)
    print("\n")
    print("Context Documents:")
    for doc in context_docs:
        print(f"- {doc.page_content}")
        print("  Metadata:", doc.metadata)
    print("\n")
    print("---"*20)
    print("\n")
    print("Response:", response)

if __name__ == "__main__":
    main()