
import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Universal PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. CSS - READABLE TEXT AND CLEAN DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main page */
    [data-testid="stAppViewContainer"] {
        background: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: #f4f7fb;
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main page text: dark and readable */
    .main,
    .main p,
    .main span,
    .main label,
    .main li,
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main strong,
    .main td,
    .main th {
        color: #202938;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #202735;
    }

    [data-testid="stSidebar"] * {
        color: #f5f7fb;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #2b3445;
        border-radius: 12px;
        padding: 10px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(120deg, #d7f7e8, #dcecff);
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 22px;
        border: 1px solid #d5e8e7;
    }

    .hero h1 {
        color: #172b3a !important;
        font-size: 2.1rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #344b5c !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Information cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
        color: #202938 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #dce5ed;
        border-radius: 14px;
        padding: 16px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #202938 !important;
    }

    /* Tables */
    [data-testid="stTable"],
    [data-testid="stDataFrame"] {
        background: #ffffff;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        color: #202938 !important;
        -webkit-text-fill-color: #202938 !important;
        background: #ffffff !important;
        caret-color: #202938 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #657386 !important;
        -webkit-text-fill-color: #657386 !important;
        opacity: 1 !important;
    }

    /* Other text inputs */
    [data-testid="stTextInput"] input {
        color: #202938 !important;
        background: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d4deea;
        min-height: 42px;
    }

    .stButton > button p {
        color: inherit !important;
    }

    /* Status messages */
    [data-testid="stAlert"] p {
        color: #202938 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 10px;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #202938 !important;
    }

    /* Divider */
    hr {
        border-color: #dce5ed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. INITIALIZE SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# 4. LOAD EMBEDDINGS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 5. LOAD GROQ MODEL
# --------------------------------------------------

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add your API key to the .env file "
            "in the same folder as app.py, then restart Streamlit."
        )
        st.stop()

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        api_key=api_key,
    )


# --------------------------------------------------
# 6. PROCESS ANY TEXT-BASED PDF
# --------------------------------------------------

def process_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Reuse the existing index if this PDF is already loaded
    if (
        st.session_state.pdf_hash == current_hash
        and st.session_state.vector_store is not None
    ):
        return

    with st.spinner("Reading your PDF and preparing it for questions..."):
        temp_path = None

        try:
            # PyPDFLoader needs a file path
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            if not pages:
                st.error("No pages were found in this PDF.")
                return

            # Split the PDF into smaller pieces
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            chunks = splitter.split_documents(pages)

            if not chunks:
                st.error(
                    "No readable text was found. "
                    "This may be a scanned PDF that needs OCR."
                )
                return

            # Create a vector store in memory
            embeddings = get_embeddings()

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="pdf_" + current_hash[:20],
            )

            st.session_state.vector_store = vector_store
            st.session_state.pdf_hash = current_hash
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.page_count = len(pages)
            st.session_state.chunk_count = len(chunks)

            # Clear previous conversation for the new PDF
            st.session_state.messages = []

        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# 7. ANSWER QUESTIONS USING RETRIEVED PDF CONTENT
# --------------------------------------------------

def ask_pdf(question):
    vector_store = st.session_state.vector_store

    if vector_store is None:
        return (
            "Please upload a PDF first.",
            [],
        )

    llm = get_llm()

    # Retrieve relevant passages
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    documents = retriever.invoke(question)

    if not documents:
        return (
            "I couldn't find relevant information in the PDF for that question.",
            [],
        )

    context = "\n\n".join(
        [
            f"Source page {doc.metadata.get('page', 0) + 1}:\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful PDF reading assistant.

Answer the user's question using the supplied PDF context.

Rules:
1. Use the PDF context as your primary source.
2. Explain the answer in clear, simple language.
3. Use bullet points when they make the answer easier to read.
4. If the answer is not present in the provided context, say:
   "I couldn't find that information in the provided PDF."
5. Do not invent facts or claim that something is in the PDF when it is not.
6. If asked to summarize, give a concise summary of the relevant content.
7. Make the answer readable with headings and bullet points when appropriate.

PDF CONTEXT:
{context}
                """,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    try:
        formatted_prompt = prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = llm.invoke(formatted_prompt)
        answer = response.content

        if not isinstance(answer, str):
            answer = str(answer)

        return answer, documents

    except Exception as error:
        return (
            f"Sorry, I couldn't generate an answer. Error: {error}",
            documents,
        )


# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 PDF Assistant")
    st.caption("Upload a document and chat with its contents.")

    st.markdown("---")

    st.markdown("### 1. Upload your document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF to ask questions about it.",
    )

    if uploaded_file is not None:
        process_pdf(uploaded_file)

    if st.session_state.vector_store is not None:
        st.success("PDF ready for questions!")

        st.caption(f"File: {st.session_state.pdf_name}")
        st.caption(f"Pages: {st.session_state.page_count}")
        st.caption(f"Text chunks: {st.session_state.chunk_count}")

    st.markdown("---")

    st.markdown("### 2. What can you ask?")

    suggestions = [
        "Summarize this document.",
        "What are the main topics?",
        "Explain the most important concepts.",
        "What are the key conclusions?",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

    st.markdown("---")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption("Your PDF index is stored in memory for this session.")


# --------------------------------------------------
# 9. MAIN PAGE
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📖 Universal PDF AI Assistant</h1>
        <p>
            Upload any text-based PDF, ask questions, understand difficult
            topics, and summarize your documents using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vector_store is None:
    st.markdown(
        """
        <div class="info-card">
            <h3>👋 Welcome!</h3>
            <p>
                Start by uploading a PDF using the sidebar on the left.
                Once your PDF is ready, ask any question in the chat box.
            </p>
            <p>
                You can use this assistant for resumes, study notes,
                reports, books, and other text-based PDF documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        f"""
        <div class="info-card">
            <strong>📄 Current document:</strong>
            {st.session_state.pdf_name}
            <br>
            <strong>📑 Pages:</strong> {st.session_state.page_count}
            &nbsp; | &nbsp;
            <strong>🧩 Text chunks:</strong> {st.session_state.chunk_count}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 10. DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View PDF sources"):
                seen = set()

                for document in message["sources"]:
                    page_number = document.metadata.get("page", 0) + 1
                    text_preview = document.page_content.strip()

                    source_key = (page_number, text_preview[:100])

                    if source_key in seen:
                        continue

                    seen.add(source_key)

                    st.markdown(f"**Page {page_number}**")
                    st.write(text_preview[:1200])

                    st.divider()


# --------------------------------------------------
# 11. CHAT INPUT AND RESPONSE
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask anything about your uploaded PDF..."
)

question = typed_question

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF in the sidebar first.")

    else:
        # Display and save the user's question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate the answer
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF and preparing an answer..."):
                answer, source_documents = ask_pdf(question)

            st.markdown(answer)

            if source_documents:
                with st.expander("📚 View PDF sources"):
                    seen = set()

                    for document in source_documents:
                        page_number = document.metadata.get("page", 0) + 1
                        text_preview = document.page_content.strip()

                        source_key = (page_number, text_preview[:100])

                        if source_key in seen:
                            continue

                        seen.add(source_key)

                        st.markdown(f"**Page {page_number}**")
                        st.write(text_preview[:1200])
                        st.divider()

        # Save answer and sources for later reruns
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_documents,
            }
        )
