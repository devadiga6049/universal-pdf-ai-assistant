
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# STEP 1: Load environment variables
load_dotenv()

# STEP 2: Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# STEP 3: Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 4: Load PDF
pdf_path = "ENVIRONMENTAL SCIENCE.pdf"

print("\nLoading PDF...")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"Total PDF pages loaded: {len(pages)}")

# Check extracted PDF text
if not pages or not any(page.page_content.strip() for page in pages):
    raise ValueError(
        "No readable text found in the PDF. "
        "Check whether it is a scanned PDF."
    )

# STEP 5: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print(f"Total text chunks created: {len(chunks)}")

# STEP 6: Create Chroma vector store
# Rebuild the collection from the current PDF
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="env_science"
)

print("Chroma vector store created successfully.")

# STEP 7: Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# STEP 8: Test PDF retrieval
question = "What are the main causes of climate change?"

docs = retriever.invoke(question)

print("\n========== RETRIEVED PDF CONTENT ==========")

for i, doc in enumerate(docs, start=1):
    print(f"\nDocument {i}:")
    print(f"Page: {doc.metadata.get('page', 'Unknown') + 1 if isinstance(doc.metadata.get('page'), int) else 'Unknown'}")
    print(doc.page_content[:1000])

# STEP 9: Create RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an assistant answering questions about
Environmental Science.

Answer the question using the provided PDF context.
Use the context carefully and explain the answer clearly.
If the context does not contain enough information,
say so instead of inventing an answer.

PDF Context:
{context}"""
    ),
    ("human", "{input}")
])

# STEP 10: Create document chain
document_chain = create_stuff_documents_chain(
    llm,
    rag_prompt
)

# STEP 11: Create retrieval chain
rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# STEP 12: Generate final answer
print("\n========== FINAL ANSWER ==========")

response = rag_chain.invoke({
    "input": question
})

print("\nQuestion:", question)
print("\nAnswer:", response["answer"])
