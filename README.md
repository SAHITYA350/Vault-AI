# Vault-AI: Your Second Brain & AI Knowledge Graph 🧠

Vault-AI is an advanced, AI-powered personal knowledge management system. It securely stores your documents (PDFs, Images), uses cutting-edge OCR and Natural Language Processing to understand their contents, and visualizes connections between different pieces of knowledge using an interactive Knowledge Graph.

## 🌟 What makes Vault-AI Unique?

Unlike standard cloud storage or basic note-taking apps, Vault-AI acts as your **Second Brain**:
- **Semantic Understanding**: It doesn't just store files; it reads and understands them using Gemini AI.
- **Automated Knowledge Graph**: It automatically extracts concepts, topics, and tags from your documents and builds a fully interactive 3D/2D visual network of your knowledge.
- **RAG Chat Assistant**: You can chat with your entire document vault. Ask a question, and Vault-AI will search through all your PDFs and images, synthesize an answer, and cite the exact pages and documents it used!
- **AI Study Lounge**: Automatically generates mock exams, flashcards, and quizzes based on the knowledge inside your documents.

## 🛠️ Technology Stack

- **Backend framework**: Django (Python) - *Chosen for its robust ORM, security features, and rapid development capabilities.*
- **Database**: SQLite / PostgreSQL - *Relational structure to maintain complex user, document, and chunk relationships.*
- **AI Engine**: Google Gemini API - *Used for highly accurate document summarization, OCR, concept extraction, and RAG (Retrieval-Augmented Generation).*
- **Frontend**: HTML5, CSS3 (Glassmorphism UI), Vanilla JavaScript.
- **Graph Visualization**: Vis.js - *A powerful, physics-based network visualization library capable of handling thousands of interconnected nodes.*
- **Authentication**: Google OAuth 2.0 (via `django-allauth`) - *Ensures secure, passwordless login.*

## 🚀 Why is it Production Ready & Scalable?

1. **Security**: We use Google OAuth. Why? Storing passwords is risky and adds friction for the user. By relying on Google's authentication infrastructure, we guarantee enterprise-grade login security and zero stored plain-text passwords.
2. **Scalability**: The processing engine is decoupled. Large PDFs are chunked into pages, and the AI processes them asynchronously. The Knowledge Graph uses optimized physics solvers (`forceAtlas2Based`) to render large datasets smoothly without crashing the browser.
3. **Storage Efficiency**: Uses a hierarchical chunking model (`Document` -> `DocumentChunk`), meaning we can retrieve only the exact paragraph needed for an AI answer instead of feeding the entire 500-page PDF, saving massive token costs.

## 📂 Folder & Architecture Structure

```text
Vault-AI/
│
├── imageuploader/          # Main Django Project Configuration
│   ├── settings.py         # Configures DB, OAuth, Templates, Static paths
│   ├── urls.py             # Global URL routing
│
├── myapp/                  # Core Application
│   ├── models.py           # Database Schema (Document, DocumentChunk, ChatMessage)
│   ├── views.py            # Route Handlers (Dashboard, Upload, Graph APIs, RAG Chat)
│   ├── ai_engine.py        # The Core AI Logic (Gemini API calls, Prompts, Chunking)
│   │
│   ├── templates/myapp/    # Frontend UI
│   │   ├── base.html       # Base layout (Sidebar, Header, Glassmorphism UI)
│   │   ├── landing.html    # Marketing, Pricing, Footer
│   │   ├── dashboard.html  # Vault overview and file uploads
│   │   ├── graph.html      # The interactive Knowledge Graph canvas
│   │   ├── chat.html       # The RAG Assistant chat interface
│   │   └── study.html      # AI generated exams and quizzes
│
├── media/                  # User Uploads (PDFs, Images) - Ignored in Git
├── db.sqlite3              # Local Database - Ignored in Git
└── .env                    # Secrets (API Keys, OAuth IDs) - Ignored in Git
```

## ⚙️ How the Code Works (The Flow)

1. **Upload Phase**: A user uploads a file (`views.py`). 
2. **Processing Phase**: The file is passed to `ai_engine.py`. If it's an image, Gemini Vision extracts the text. If it's a PDF, PyMuPDF extracts text page-by-page.
3. **Chunking & Concept Extraction**: `ai_engine.py` breaks the text into smaller chunks, asks Gemini to summarize it, and generate "Concept Labels" and "Tags".
4. **Graph Generation**: When the user opens the Knowledge Graph (`graph.html`), a fetch request calls `graph_data_api()`. This function translates the SQL database relationships into Nodes and Edges for Vis.js to draw.
5. **RAG Chat**: When a user asks a question in `chat.html`, `ai_engine.py` does a semantic keyword search across the user's `DocumentChunk`s, finds the most relevant paragraphs, and feeds them to Gemini to generate a synthesized answer with citations.

---
*Built with passion by Sahitya Ghosh. © 2026 AI Study Lounge.*
