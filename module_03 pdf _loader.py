from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")

pdf_path = r".\ENVIRONMENTAL SCIENCE.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

# Print number of pages
print(f"Number of pages: {len(pages)}")

# Print first 500 characters
print(pages[0].page_content[:500])

# Print metadata of page 1
print(f"\nMetadata of page 1: {pages[0].metadata}")

# Calculate total characters
total_characters = sum(len(page.page_content) for page in pages)

print(f"\nTotal characters present in PDF: {total_characters:,}")

# Estimate tokens
estimated_tokens = total_characters // 4

print(f"So estimated tokens are: {estimated_tokens:,}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
    )

chunks = text_splitter.split_documents(pages)

print(f"\nFirst chunk content:\n {chunks[0].page_content}")
print(f"\n Total chunks created after splitting: {len(chunks)}")

print(chunks[0].page_content)