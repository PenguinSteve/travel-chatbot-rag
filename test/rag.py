from langchain_groq import ChatGroq
from ingestion.store_data import connect_to_pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_dotenv()


def get_retriever(index_name: str, k: int = 5):
    vector_store = connect_to_pinecone(index_name=index_name)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    print(f"\n---------------------Created retriever with top {k} documents---------------------\n")
    return retriever

def generate_groq_response(retriever, query: str):
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    llm = ChatGroq(model=llm_model, temperature=0, api_key=groq_api_key)

    system = """You are an AI assistant that helps people find information about tourism.
    You are given the following extracted parts of a long document and a question.
    Provide a conversational answer based on the context provided.xw
    If you don't know the answer, just say "I don't know". Don't try to make up an answer.
    Always answer in Vietnamese.
    """
    
   

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    print("\n---------------------Generating response...---------------------\n")
    

    # start_time_retrieval = os.times()
    # print("\n---------------------Retrieving relevant documents...---------------------\n")
    context_docs = retriever.invoke(query)
    end_time_retrieval = os.times()
    # print("\n---------------------Retrieved relevant documents in", end_time_retrieval.user - start_time_retrieval.user, "seconds.---------------------\n")

    if not context_docs.strip():
        prompt_input = {
            "context": "No relevant documents found. Please answer based on the data general",
            "question": query
        }
    else:
        prompt_input = {
            "context": "\n\n".join([doc.page_content for doc in context_docs]),
            "question": query
        }

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(prompt_input)
    return response, context_docs

def main():
    start_time = os.times()
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-tourism")
    retriever = get_retriever(index_name=index_name, k=5)


    query = "Hi, I'm a foreigner planning to visit Vietnam. Can you suggest some must-visit tourist attractions and the best time to visit them?"
    response, context_docs = generate_groq_response(retriever, query)

    print("\n---------------------Context Documents:---------------------\n")
    for index, doc in enumerate(context_docs):
        if index > 0:
            print("--------------------------------------------------------------\n")
        print(f"Context number {index}:\n {doc.page_content}")
        print("  Metadata:", doc.metadata)
    print("\n---------------------End of Context Documents---------------------\n")
    print("Question:", query, "\n")
    print("Response:", response)
    end_time = os.times()
    print("\n---------------------Time taken:", end_time.user - start_time.user, "seconds---------------------")

if __name__ == "__main__":
    main()