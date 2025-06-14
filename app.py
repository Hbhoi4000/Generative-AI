# PDF SUMMARIZE

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

print(google_api_key) 


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a vector store for text chunks."""
    try:
        # Convert text chunks into Document objects
        documents = [Document(page_content=chunk) for chunk in text_chunks]

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create the vector store
        vector_store = FAISS.from_documents(documents, embeddings)

        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
def get_conversational_chain(db):
    """Set up the conversational chain with a custom prompt."""
    prompt_template = """
    You are an AI assistant designed to help users interact with PDF documents. Your responsibilities include:
    
    1. Extracting and summarizing the key sections or topics from the uploaded PDF(s).
    2. Answering user questions based solely on the content of the uploaded PDF(s).
    3. If the question cannot be answered using the PDF content, respond with: 
       "I'm sorry, but the information you requested is not available in the uploaded document(s)."

    Guidelines for interaction:
    - Maintain context across multiple user questions.
    - Refer to the same PDF(s) unless new ones are uploaded.
    - Provide clear, concise, and accurate answers supported by the PDF content.

    Previous conversation history:
    {messages}

    Relevant context from the document(s):
    {context}

    User's question:
    {input}

    Your answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","input","messages"])
    combine_docs_chain = create_stuff_documents_chain(llm=model,prompt=prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def user_input(user_question,db):
    """Process user input and generate a response."""
    chain = get_conversational_chain(db)
    response = chain.invoke({"input": user_question, "messages": st.session_state.messages})
    #response = llm.invoke(user_question)
    return response

def reset_conversation():
    """Reset the conversation state."""
    for key in st.session_state.keys():
        del st.session_state[key]

# Streamlit App
st.set_page_config("Chat PDF")
st.header(" pdf Talk with Bot!")

# Sidebar
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    st.button('Reset Chat', on_click=reset_conversation)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process uploaded PDFs
if pdf_docs:
    raw_text = get_pdf_text(pdf_docs)
    print(raw_text)
    text_chunks = get_text_chunks(raw_text)
    print(text_chunks)
    db=get_vector_store(text_chunks)
    

# Handle user input
user_question = st.chat_input("Ask a Question from the PDF Files")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user").write(user_question)
    response = user_input(user_question,db)
    output = response
    st.session_state.messages.append({"role": "assistant", "content": output['answer']})
    st.chat_message("assistant").write(str(output['answer']))
