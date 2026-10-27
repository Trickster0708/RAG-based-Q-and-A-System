import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

CHROMA_DIR = "chroma-db"

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its content.")


# --------------------------------------------------
# Initialize Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

def create_vector_database(pdf_path):

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Remove old database
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vectorstore, len(docs), len(chunks)


# --------------------------------------------------
# Load Vector Database
# --------------------------------------------------

def load_vector_database():

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return ChatMistralAI(
        model="mistral-small-2506"
    )


llm = load_llm()


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use ONLY the provided context to answer the question.

            If the answer is not present in the context, say:
            "I could not find the answer in the document."

            Do not make up information.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar - PDF Upload
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.write(f"**Selected:** {uploaded_file.name}")

        if st.button("Process PDF", use_container_width=True):

            with st.spinner(
                "Loading PDF, creating chunks and generating embeddings..."
            ):

                # Temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())

                    temp_pdf_path = temp_file.name

                try:

                    vectorstore, page_count, chunk_count = (
                        create_vector_database(temp_pdf_path)
                    )

                    st.session_state.vectorstore = vectorstore

                    st.session_state.pdf_name = uploaded_file.name

                    st.session_state.messages = []

                    st.success("PDF processed successfully!")

                    st.info(
                        f"Pages: {page_count}\n\n"
                        f"Chunks: {chunk_count}"
                    )

                finally:

                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)


# --------------------------------------------------
# Display Current PDF
# --------------------------------------------------

if st.session_state.pdf_name:

    st.success(
        f"Currently loaded PDF: "
        f"**{st.session_state.pdf_name}**"
    )


# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

query = st.chat_input(
    "Ask a question about your PDF..."
)


if query:

    # Check if PDF exists
    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # Retriever
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(final_prompt)

            answer = response.content

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )