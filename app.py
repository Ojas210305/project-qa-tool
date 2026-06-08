from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import docx
import pandas as pd
import requests
import io
import os
import threading
import time
import uuid
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
COHERE_KEY     = os.environ.get("COHERE_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── COHERE EMBEDDING ─────────────────────────────────────────

def get_embedding(text, input_type="search_document"):
    try:
        response = requests.post(
            "https://api.cohere.com/v1/embed",
            headers={
                "Authorization": f"Bearer {COHERE_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "texts": [text[:512]],
                "model": "embed-english-v3.0",
                "input_type": input_type
            }
        )
        result = response.json()
        if "embeddings" in result:
            return result["embeddings"][0]
        else:
            print(f"Cohere error: {result}")
            return None
    except Exception as e:
        print(f"Cohere embedding error: {e}")
        return None


# ── TEXT EXTRACTION ──────────────────────────────────────────

def extract_text_from_file(file):
    filename = file.filename.lower()
    content = ""
    try:
        if filename.endswith(".txt") or filename.endswith(".md"):
            content = file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            file_bytes = file.read()
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf:
                content += page.get_text()
            pdf.close()
        elif filename.endswith(".docx"):
            file_bytes = file.read()
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                content += para.text + "\n"
        elif filename.endswith(".csv"):
            file_bytes = file.read()
            df = pd.read_csv(io.BytesIO(file_bytes))
            content = df.to_string(index=False)
        else:
            content = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        content = f"Could not read file: {str(e)}"
    return content.strip()


# ── CHUNKING ─────────────────────────────────────────────────

def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def home():
    return "Project QA Tool Backend is running!"


@app.route("/create_project", methods=["POST"])
def create_project():
    data = request.json
    name        = data.get("name", "")
    description = data.get("description", "")
    date        = data.get("date", "")

    if not name:
        return jsonify({"error": "Project name required"}), 400

    try:
        result = supabase.table("projects").insert({
            "name": name,
            "description": description,
            "date": date
        }).execute()
        return jsonify({"project": result.data[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_projects", methods=["GET"])
def get_projects():
    try:
        result = supabase.table("projects").select("*").order("created_at", desc=True).execute()
        return jsonify({"projects": result.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_project", methods=["POST"])
def delete_project():
    data = request.json
    project_id = data.get("project_id", "")
    try:
        supabase.table("projects").delete().eq("id", project_id).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    files      = request.files.getlist("files")
    project_id = request.form.get("project_id", "")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    if not project_id:
        return jsonify({"error": "Project ID required"}), 400

    uploaded = []

    for file in files:
        filename = file.filename
        text     = extract_text_from_file(file)
        if not text:
            continue

        chunks  = chunk_text(text, chunk_size=1000, overlap=100)
        records = []

        for chunk in chunks:
            embedding = get_embedding(chunk, input_type="search_document")
            if embedding:
                records.append({
                    "project_id": project_id,
                    "file_name":  filename,
                    "content":    chunk,
                    "embedding":  embedding
                })

        if records:
            supabase.table("chunks").insert(records).execute()
            uploaded.append({"name": filename, "chunks": len(records)})

    return jsonify({"uploaded": uploaded})


@app.route("/get_files", methods=["POST"])
def get_files():
    data       = request.json
    project_id = data.get("project_id", "")
    try:
        result = supabase.table("chunks").select("file_name").eq("project_id", project_id).execute()
        files  = list(set([r["file_name"] for r in result.data]))
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_file", methods=["POST"])
def delete_file():
    data       = request.json
    project_id = data.get("project_id", "")
    file_name  = data.get("file_name", "")
    try:
        supabase.table("chunks").delete().eq("project_id", project_id).eq("file_name", file_name).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data         = request.json
    question     = data.get("question", "")
    project_id   = data.get("project_id", "")
    project_name = data.get("project_name", "")
    project_desc = data.get("project_desc", "")
    file_name    = data.get("file_name", None)
    history      = data.get("history", [])

    if not question:
        return jsonify({"error": "No question provided"}), 400

    query_embedding = get_embedding(question, input_type="search_query")
    if not query_embedding:
        return jsonify({"error": "Could not generate embedding for question"}), 500

    try:
        if file_name:
            result = supabase.rpc("match_chunks", {
                "query_embedding":  query_embedding,
                "match_project_id": project_id,
                "match_file_name":  file_name,
                "match_count":      5
            }).execute()
        else:
            result = supabase.rpc("match_chunks_all", {
                "query_embedding":  query_embedding,
                "match_project_id": project_id,
                "match_count":      5
            }).execute()

        matched_chunks = result.data
        print(f"Question: {question}")
        print(f"Matched chunks: {len(matched_chunks)}")
        for chunk in matched_chunks:
            print(f"  - {chunk['file_name']}: {chunk['content'][:100]}")

    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)}"}), 500

    if not matched_chunks:
        return jsonify({"reply": "I couldn't find relevant information in the documents for your question.", "tokens_used": 0})

    context = ""
    for chunk in matched_chunks:
        context += f"\n--- From: {chunk['file_name']} ---\n"
        context += chunk['content']
        context += "\n"

    if file_name:
        system_msg = f"""You are an AI assistant for the project "{project_name}".
You are focused on the file: {file_name}

Here are the most relevant sections from this file:

{context}

Instructions:
1. Answer based ONLY on the provided sections.
2. Be precise and helpful.
3. If the answer is not in the sections, say so clearly."""
    else:
        system_msg = f"""You are an AI assistant for the project "{project_name}".
Project description: {project_desc}

Here are the most relevant sections from the project documents:

{context}

Instructions:
1. Answer based ONLY on the provided sections.
2. Always mention WHICH FILE the information comes from.
3. If asked to compare files, compare them clearly.
4. If the answer is not in the sections, say so clearly.
5. Format comparisons as bullet points or tables."""

    messages = [{"role": "system", "content": system_msg}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type":  "application/json"
            },
            json={
                "model":      "openrouter/free",
                "messages":   messages,
                "max_tokens": 1000
            }
        )
        result = response.json()
        if "error" in result:
            return jsonify({"error": result["error"]["message"]}), 500

        reply = result["choices"][0]["message"]["content"]

        # Get token usage from OpenRouter response
        tokens_used = 0
        if "usage" in result:
            tokens_used = result["usage"].get("total_tokens", 0)

        return jsonify({"reply": reply, "tokens_used": tokens_used})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CHAT HISTORY WITH SESSION SUPPORT ────────────────────────

@app.route("/save_message", methods=["POST", "OPTIONS"])
def save_message():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.json
    project_id = data.get("project_id")
    role       = data.get("role")
    content    = data.get("content")
    session_id = data.get("session_id")
    tokens     = data.get("tokens_used", 0)
    try:
        supabase.table("chat_history").insert({
            "project_id": project_id,
            "role":       role,
            "content":    content,
            "session_id": session_id,
            "is_active":  True,
            "tokens_used": tokens
        }).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_history", methods=["POST", "OPTIONS"])
def get_history():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.json
    project_id = data.get("project_id")
    session_id = data.get("session_id", None)
    try:
        query = supabase.table("chat_history").select("*")\
            .eq("project_id", project_id)\
            .eq("is_active", True)

        if session_id:
            query = query.eq("session_id", session_id)

        result = query.order("created_at").execute()
        return jsonify({"history": result.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/new_chat", methods=["POST", "OPTIONS"])
def new_chat():
    """Soft delete - marks old messages as inactive, returns new session ID"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.json
    project_id = data.get("project_id")
    try:
        # Mark all current messages as inactive (soft delete)
        supabase.table("chat_history")\
            .update({"is_active": False})\
            .eq("project_id", project_id)\
            .eq("is_active", True)\
            .execute()

        # Generate new session ID
        new_session_id = str(uuid.uuid4())
        return jsonify({"success": True, "session_id": new_session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_token_usage", methods=["POST", "OPTIONS"])
def get_token_usage():
    """Get token usage stats for a project"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.json
    project_id = data.get("project_id")
    try:
        result = supabase.table("chat_history")\
            .select("role, content, tokens_used, created_at, session_id")\
            .eq("project_id", project_id)\
            .eq("role", "user")\
            .order("created_at", desc=True)\
            .execute()
        return jsonify({"usage": result.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── KEEP ALIVE ───────────────────────────────────────────────

def keep_alive():
    while True:
        time.sleep(840)
        try:
            requests.get("https://project-qa-tool.onrender.com")
            print("Keep alive ping sent")
        except:
            pass


if __name__ == "__main__":
    t = threading.Thread(target=keep_alive)
    t.daemon = True
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)