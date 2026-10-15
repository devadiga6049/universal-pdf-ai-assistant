
# Universal PDF AI Assistant 📚

An AI-powered PDF question-answering application built with Python, Streamlit, LangChain, Groq, Hugging Face embeddings, and Chroma.

## Features

- Upload text-based PDF documents
- Ask questions about document content
- Generate summaries and explanations
- Retrieve relevant passages from PDFs
- View source passages and page numbers
- Interactive chat interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- Hugging Face Sentence Transformers
- Chroma vector database
- PyPDF

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Get a key from https://console.groq.com/keys.

Never upload your actual API key to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Note

The app uses an in-memory vector store. PDFs need to be processed again in a new session. Scanned PDFs may require OCR.

## Author

Add your name and portfolio links here.
