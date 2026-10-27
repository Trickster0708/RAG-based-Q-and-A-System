# 📚 RAG-based Q&A System

A **Retrieval-Augmented Generation (RAG)** system built with LangChain, ChromaDB, HuggingFace Embeddings, and MistralAI. Comes with both a **Streamlit web UI** and a **CLI interface** for asking questions about PDF documents.

---

## ✨ Features

- 📄 **PDF ingestion** — Load and chunk PDF documents automatically
- 🔍 **Semantic search** — HuggingFace `all-MiniLM-L6-v2` embeddings stored in ChromaDB
- 🤖 **MistralAI LLM** — Powered by `mistral-small-2506` for accurate, context-grounded answers
- 🔁 **MMR Retrieval** — Maximal Marginal Relevance for diverse, relevant document chunks
- 💬 **Chat history** — Full conversation memory in the Streamlit UI
- 🧪 **Retriever experiments** — Example scripts for Similarity, MMR, and MultiQuery retrievers

---

## 🗂️ Project Structure

```
RAG/
├── app.py                  # Streamlit web UI (main app)
├── main.py                 # CLI-based RAG interface
├── create_database.py      # One-time script to build ChromaDB from a PDF
├── requirement.txt         # Python dependencies
│
├── Document loaders/       # PDF loading experiments & sample documents
│   ├── pdf.py
│   ├── page.py
│   └── test.py
│
├── retrievers/             # Retriever strategy examples
│   ├── mmr.py              # MMR vs Similarity comparison
│   ├── multiquery.py       # MultiQueryRetriever example
│   └── arixv.py            # ArXiv retriever experiment
│
└── vector store/           # Vector store setup examples
    └── DB.py
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Trickster0708/RAG-based-Q-A-System.git
cd RAG-based-Q-A-System
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 🖥️ Usage

### Option A — Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
1. Upload any PDF from the sidebar
2. Click **Process PDF**
3. Ask questions in the chat box!

### Option B — Command Line Interface
First, build the database from a PDF:
```bash
python create_database.py
```
Then start the CLI chat:
```bash
python main.py
```

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Framework | [LangChain](https://www.langchain.com/) |
| LLM | [MistralAI](https://mistral.ai/) (`mistral-small-2506`) |
| Embeddings | [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Web UI | [Streamlit](https://streamlit.io/) |
| PDF Loader | LangChain `PyPDFLoader` |

---

## ⚙️ How It Works

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings → ChromaDB
                                                                                    ↓
User Query → Embeddings → MMR Retriever → Relevant Chunks → MistralAI LLM → Answer
```

---

## 📝 Notes

- The `.env` file and `chroma-db/` directory are excluded from version control
- Large PDFs may take a moment to process during embedding generation
- The MMR retriever (`lambda_mult=0.5`) balances relevance and diversity in retrieved chunks
