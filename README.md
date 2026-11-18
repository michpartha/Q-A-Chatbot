# Research Paper Q&A

A Streamlit web app that lets you ask questions about a collection of research papers (PDFs). It uses a Retrieval-Augmented Generation (RAG) pipeline: the PDFs are split into chunks, embedded into a FAISS vector store, and the most relevant chunks are passed to a local LLM (served by Ollama) to generate an answer grounded in the papers.

## Features

- Loads all PDFs from a local folder (`research_papers/`)
- Splits documents into overlapping text chunks for better retrieval
- Builds a FAISS vector index using Ollama embeddings
- Answers questions using only the retrieved context (no outside knowledge)
- Shows the source chunks used for each answer in an expandable panel
- Prints the response time to the terminal

## Tech Stack

- [Streamlit](https://streamlit.io/) – web UI
- [LangChain](https://www.langchain.com/) – orchestration, prompts, document loading, splitting
- [Ollama](https://ollama.com/) – local LLM (`qwen2.5:latest`) and embeddings (`nomic-embed-text`)
- [FAISS](https://github.com/facebookresearch/faiss) – vector similarity search
- [PyPDF](https://pypdf.readthedocs.io/) – PDF text extraction

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── research_papers/    # Folder containing the PDF files to index
├── .env                # Environment variables (optional)
└── README.md
```

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally
- The following Ollama models pulled:

```bash
ollama pull qwen2.5:latest
ollama pull nomic-embed-text
```

## Installation

1. Clone or download the project and move into the folder:

   ```bash
   cd your-project-folder
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install the dependencies:

   ```bash
   pip install streamlit langchain langchain-community langchain-ollama langchain-text-splitters langchain-core faiss-cpu pypdf python-dotenv
   ```

4. Create a `research_papers/` folder in the project root and add your PDF files to it.

5. (Optional) Create a `.env` file if you plan to use an Ollama API key:

   ```env
   OLLAMA_API_KEY=your_key_here
   ```

## Usage

1. Make sure Ollama is running (`ollama serve`, or the Ollama app is open).

2. Start the app:

   ```bash
   streamlit run app.py
   ```

3. In the browser window that opens:
   - Click **Document Embedding** to load and index the PDFs in `research_papers/`. Wait for the "Vector database is ready" message.
   - Type your question in the text box and press Enter.
   - Read the answer, and expand **Document similarity Search** to see the source passages used.

> **Note:** You must click **Document Embedding** before asking questions. The index is kept in the session, so if you refresh the browser page you will need to build it again.

## How It Works

1. **Ingestion** – `PyPDFDirectoryLoader` reads every PDF in `research_papers/`.
2. **Chunking** – `RecursiveCharacterTextSplitter` splits the text into 1000-character chunks with a 200-character overlap.
3. **Embedding & indexing** – `nomic-embed-text` converts each chunk into a vector, which is stored in FAISS.
4. **Retrieval** – When you ask a question, the retriever finds the chunks most similar to it.
5. **Generation** – The chunks are inserted into a prompt, and `qwen2.5` produces an answer based only on that context.

## Configuration

You can change these values in `app.py`:

| Setting | Location | Default | Description |
|---|---|---|---|
| LLM model | `ChatOllama(model=...)` | `qwen2.5:latest` | Ollama model used to generate answers |
| Embedding model | `OllamaEmbeddings(model=...)` | `nomic-embed-text` | Model used to create embeddings |
| Papers folder | `PyPDFDirectoryLoader(...)` | `research_papers` | Directory containing PDFs |
| Chunk size | `chunk_size` | `1000` | Characters per chunk |
| Chunk overlap | `chunk_overlap` | `200` | Overlap between consecutive chunks |
| Temperature | `ChatOllama(temperature=...)` | `0` | Lower = more deterministic answers |

## Switching to Groq

The code includes a commented-out Groq option. To use it:

1. Install `langchain-groq` and add `GROQ_API_KEY` to your `.env` file.
2. Uncomment `from langchain_groq import ChatGroq`.
3. Replace the `ChatOllama(...)` line with your `ChatGroq(...)` configuration.

## Troubleshooting

- **"Click 'Document Embedding' first."** – Build the vector store before asking a question.
- **Connection errors to Ollama** – Make sure Ollama is running (`ollama serve`) and the models are pulled.
- **No documents loaded / empty index** – Check that `research_papers/` exists and contains PDF files. Scanned PDFs without text will produce empty chunks; OCR them first.
- **Slow embedding** – Large collections take time on the first run. Reduce the number of PDFs or use a smaller embedding model.
- **Answers say the information is not in the context** – Try a more specific question, or increase `chunk_size` / the number of retrieved documents.

## Possible Improvements

- Save the FAISS index to disk so it doesn't need to be rebuilt on every run
- Show which PDF and page each chunk came from
- Add a chat history for follow-up questions
- Use `st.cache_resource` instead of session state for the vector store

## License

Add your license here.
