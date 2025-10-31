from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.llm import llm_chat
from app.repositories.chat_repository import ChatRepository
from app.models.chat_schema import ChatMessage


def chat_history_to_messages(retriever: any, question: str, session_id: str, chat_repo: ChatRepository, chat_history: list):
   
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    print('---> Chat History:', chat_history)
    # Create a ChatPromptTemplate for contextualizing the question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),  # Set the system prompt
            MessagesPlaceholder("chat_history"),  # Placeholder for the chat history
            ("human", "{input}"),  # Placeholder for the user's input question
        ]
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm_chat(),  # Pass the language model instance
        retriever=retriever,  # Pass the retriever instance
        prompt=contextualize_q_prompt  # Pass the prompt for contextualizing the question
        # contextualize_q_prompt=contextualize_q_prompt  # Pass the prompt for contextualizing the question
    )
    
    
    docs = history_aware_retriever.invoke(
        {"input": question, "chat_history": chat_history}  # Example input with empty chat history
    )
   
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""
    You are a helpful and knowledgeable travel assistant.
    
    Conversation so far:
    {chat_history}

    User question:
    "{question}"

    Here are the relevant information snippets retrieved from the travel knowledge base:
    {context}

    Your task:
    - Read the retrieved information carefully.
    - Provide a concise, informative, and engaging answer to the user's question.
    - Only use facts supported by the retrieved content.
    - If some details are missing, say so politely instead of making up information.

    Now, write your final answer in clear and natural English (or Vietnamese if the user’s question was in Vietnamese).
    """
    
    final_answer = llm_chat().invoke(prompt)
    chat_repo.save_message(session_id=session_id, message=ChatMessage(content=final_answer.content, role="ai"))
    
    
    return {"message": question, "docs": [d.page_content for d in docs], "final_answer": final_answer.content}