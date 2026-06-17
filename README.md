# Project QA Tool

An AI-powered Document Question & Answer system that enables users to upload documents, organize them into projects, and ask natural language questions about their content.

The application uses **Retrieval-Augmented Generation (RAG)** to generate answers grounded in uploaded documents rather than relying solely on an LLM's general knowledge. Semantic search is powered by **Cohere embeddings**, responses are generated via **OpenRouter**, and document vectors are stored in **Supabase (pgvector)**.

---

## Key Features

### Project-Based Document Management

Organize related files into dedicated projects and search across all documents within a project.

### Multi-Format Document Support

Supports:

* PDF
* DOCX
* CSV
* TXT
* Markdown (.md)

### Retrieval-Augmented Generation (RAG)

Questions are answered using context retrieved from uploaded documents, improving accuracy and reducing hallucinations.

### Advanced PDF Table Extraction

Unlike text-only extraction approaches, tables are detected directly from PDF layout information.

Benefits include:

* Accurate table preservation
* Multi-page table merging
* Better retrieval of large reference tables
* Reduced context fragmentation

### Flexible Question Answering

Ask questions:

* Across all files in a project
* Against a specific file only

### Chat History & Session Management

Starting a new conversation does not permanently remove previous chats.

Instead:

* Previous conversations are archived
* Session IDs are maintained
* Historical chats remain retrievable

### Token Usage Tracking

Monitor token consumption per query for usage analytics and cost awareness.

---

## Architecture Overview

```text
Document Upload
       │
       ▼
Document Parsing
(PDF, DOCX, CSV, TXT)
       │
       ▼
Chunking & Table Processing
       │
       ▼
Cohere Embeddings
       │
       ▼
Supabase Vector Storage
       │
       ▼
User Question
       │
       ▼
Similarity Search
       │
       ▼
Relevant Context Retrieval
       │
       ▼
OpenRouter LLM
       │
       ▼
Grounded Answer
```

---

## Technology Stack

| Layer          | Technology                    |
| -------------- | ------------------------------ |
| Backend        | Flask (Python)                |
| Embeddings     | Cohere (`embed-english-v3.0`) |
| LLM            | OpenRouter                    |
| Database       | Supabase (PostgreSQL)         |
| Vector Store   | pgvector                      |
| PDF Processing | PyMuPDF (`fitz`)              |
| Frontend       | HTML, CSS, JavaScript         |
| Deployment     | Render                        |

---

## How It Works

### 1. Upload & Process

When a document is uploaded:

* Text is extracted
* Tables are detected directly from PDF layout data
* Multi-page tables are merged
* Text content is chunked with overlap for contextual continuity

### 2. Generate Embeddings

Each chunk is embedded using Cohere and stored in Supabase together with:

* Project ID
* File name
* Chunk content
* Vector embedding

### 3. Retrieve Relevant Content

When a question is asked:

* The query is embedded
* Vector similarity search is performed using:

  * `match_chunks`
  * `match_chunks_all`
* Relevant chunks are retrieved

### 4. Generate Grounded Answers

Retrieved context is passed to an LLM through OpenRouter.

The model generates an answer based on the retrieved document content and identifies the source file(s) used.

---

## Project Structure

```text
project_qa_tool/
│
├── app.py
│   Flask backend
│   - document extraction
│   - chunking
│   - embeddings
│   - vector search
│   - chat history
│
├── index.html
│   Frontend interface
│   - project management
│   - file uploads
│   - chat UI
│
├── requirements.txt
│   Python package dependencies
│
└── .env
    Environment variables
```

---

## Setup

### Environment Variables

This project requires API keys from OpenRouter, Supabase, and Cohere. You'll need to create your own accounts and generate your own keys for each — they are not shared or bundled with this repo.

Create a `.env` file in the project root:

```env
OPENROUTER_KEY=your_openrouter_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_key
COHERE_KEY=your_cohere_api_key
```

Where to get each key:

* **OPENROUTER_KEY** — sign up at [openrouter.ai](https://openrouter.ai), generate an API key from your account dashboard
* **SUPABASE_URL** / **SUPABASE_KEY** — create a project at [supabase.com](https://supabase.com), then find the project URL and service role key under Project Settings → API
* **COHERE_KEY** — sign up at [cohere.com](https://cohere.com), generate an API key from your dashboard

If deploying to Render (or any other host), add these same variables under your service's **Environment** settings rather than relying on the local `.env` file — the `.env` file is only read when running locally.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Default server:

```text
http://localhost:5000
```

---

## API Endpoints

| Endpoint           | Method | Description                                  |
| ------------------ | ------ | --------------------------------------------- |
| `/create_project`  | POST   | Create a project                             |
| `/get_projects`    | GET    | Retrieve all projects                        |
| `/delete_project`  | POST   | Delete a project                             |
| `/upload`          | POST   | Upload and process documents                 |
| `/get_files`       | POST   | Retrieve project files                       |
| `/delete_file`     | POST   | Delete a file and associated chunks          |
| `/ask`             | POST   | Ask a question against a file or project     |
| `/save_message`    | POST   | Save a chat message                          |
| `/get_history`     | POST   | Retrieve chat history                        |
| `/new_chat`        | POST   | Archive current chat and start a new session |
| `/get_token_usage` | POST   | Retrieve token usage statistics              |

---

## Current Limitations

### Multi-Page Lists

Tables spanning multiple pages are handled correctly.

Numbered and bulleted lists that continue across pages may still be split into separate chunks.

**Planned enhancement:**

* Sequential list detection
* Font and indentation analysis
* Automatic list merging

### No OCR Support

Text embedded within images is not currently extracted.

Potential future integration:

* Tesseract OCR
* OCR preprocessing pipeline

### Free-Tier Hosting Constraints

Render free instances may enter a sleep state after inactivity.

A keep-alive mechanism is included to reduce cold-start delays.

---

## Future Improvements

* OCR support for scanned PDFs
* Better handling of multi-page lists
* Document citations with page references
* User authentication
* Streaming responses
* Multi-user collaboration
* Hybrid keyword + vector search
* Reranking for improved retrieval quality

---

## Author

**Ojas**

A learning project exploring:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Document Processing Pipelines
* LLM Application Development
