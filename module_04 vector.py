
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
# Give the correct path to your PDF file
pdf_path = "ENVIRONMENTAL SCIENCE.pdf"

print(f"Loading PDF from path: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"PDF loaded successfully. Total pages: {len(pages)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(pages)

print(f"Total chunks created: {len(chunks)}")

embeddings =HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

sample_embedding = embeddings.embed_query("Hello, world!")
#print(f"embedding dimension: {len(sample_embedding)}")
#print(f"First 5 values: {sample_embedding[:5]}")

persist_directory = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="environmental_science",
)

print(f"Stored{len(chunks)} Chunks in ChromaDB!")
print(f"Data saved to :{os.path.abspath(persist_directory)}")

query = "What are the effects on water pollution ?"

results = vector_store.similarity_search(query, k=3)
for doc in results :
    print(f"{doc.page_content[:200]}...")
    print()