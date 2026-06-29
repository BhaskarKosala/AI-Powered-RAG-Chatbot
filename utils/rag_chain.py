from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def load_vector_store(embedding_model):
    '''Loading saved FAISS vector database'''
    vector_store = FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store

def retrive_documents(vector_store,query):
    '''Retrieve top 3 similar chunks'''
    documents = vector_store.similarity_search(
        query,
        k=3
    )
    return documents

def generate_response(question,retrieved_docs):
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = PromptTemplate(
        input_variables=["context","question"],
        template="""
You are a helpful AI assistant.
Answer only from the provided context - do not use outside knowledge
Include relevant details, numbers, and explanations to give a thorough response.
Summarize long information, ideally in bullets where needed.
If the answer is not available in the context, reply: "I couldn't find that information in the uploaded PDF.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    final_prompt = prompt.format(
        context = context,
        question = question
    )

    llm = ChatOllama(
        model="llama3"
    )

    response = llm.invoke(final_prompt)

    return response.content