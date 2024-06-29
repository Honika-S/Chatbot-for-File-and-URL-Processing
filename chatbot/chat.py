import streamlit as st
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from docx import Document
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="💬 Chatbot")

hf_api_token = os.getenv("HF_API_TOKEN")

# Define a function to load and split a PDF document into chunks
def load_doc(list_file_path):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Define a function to read the text content of a PDF file
def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define a function to read the text content of a DOCX file
def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Define a function to read the text content of a web page
def read_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the main content
    text = " ".join([p.text for p in soup.find_all('p')])
    return text

# Define a function to split the text into chunks
def split_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(docs)

# Define a function to create a FAISS vector database from text chunks
def create_db(splits):
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = FAISS.from_texts(splits, embeddings_model)
    return vectordb

# Define a function to initialize the LLM chain
def initialize_llmchain(vector_db, temperature, max_tokens, top_k):
    llm_model = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Hard-coded to Llama model

    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        huggingfacehub_api_token=hf_api_token,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )

    retriever = vector_db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

# Define a function to generate a response using the LLM
def generate_llama2_response(prompt_input):
    qa_chain = initialize_llmchain(st.session_state.vector_db, temperature, max_tokens, top_k)
    response = qa_chain({"query": prompt_input})
    return response["result"]

# Define a function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Initialize session state variables
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "url" not in st.session_state:
    st.session_state.url = None

# Main app layout
st.title("QA Chatbot with File and URL Upload")
st.markdown("---")

# Sidebar for model settings, file upload, and URL input
with st.sidebar:

    st.sidebar.subheader("Upload a Document or Enter a URL")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    url = st.sidebar.text_input("Or enter a web page URL")

    # Check if a new file or URL has been uploaded or entered
    if uploaded_file != st.session_state.uploaded_file or url != st.session_state.url:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.url = url
        clear_chat_history()
        st.session_state.vector_db = None

    if uploaded_file is not None and st.session_state.vector_db is None:
        with st.spinner("Converting to Vectors..."):
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = read_docx(uploaded_file)
            chunks = split_chunks(docs=text)
            st.session_state.vector_db = create_db(chunks)
            st.sidebar.markdown('<p style="color:green;">Document processed and vector database created!</p>', unsafe_allow_html=True)

    elif url and st.session_state.vector_db is None:
        with st.spinner("Converting to Vectors..."):
            text = read_web_page(url)
            chunks = split_chunks(docs=text)
            st.session_state.vector_db = create_db(chunks)
            st.sidebar.markdown('<p style="color:green;">Web page processed and vector database created!</p>', unsafe_allow_html=True)

    st.sidebar.title("Model Settings")
    # Model parameters (for Llama only)
    temperature = st.sidebar.slider('Temperature', 0.1, 1.0, 0.1)
    top_k = st.sidebar.slider('Top_k', 1, 10, 1)
    max_tokens = st.sidebar.slider('Max Tokens', 1, 512, 1024)

if st.session_state.vector_db is not None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input(disabled=not hf_api_token):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = response
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

else:
    st.write("Please upload a document or enter a URL to initialize the database.")

# Button to clear chat history
if len(st.session_state.messages) > 1 and st.button('Clear Chat History'):
    clear_chat_history()
    st.experimental_rerun()


