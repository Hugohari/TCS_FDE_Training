"""
IT Support Chat Bot - core logic
    1. RAG      : VoyageAI embeddings (voyage-3.5-lite) + in-memory vector DB over
                  it_sector.txt. Runs for every question; the matching FAQ sections
                  are attached to the user message.
    2. Tools    : web_search (server tool, runs on Anthropic's side)
                  get_current_datetime (client tool, runs here)
    3. MCP      : create_support_ticket / check_ticket_status from mcp_server.py
"""

import asyncio
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import Client

load_dotenv(override=True)  # automatically loading the .env variables

client = Anthropic()
voyage_client = voyageai.Client()
model = "claude-haiku-5-5"
embedding_model = "voyage-3.5-lite"  # Voyage's cheapest embedding model
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
BASE_DIR = Path(__file__).parent
EMBEDDING_CACHE = BASE_DIR / "faq_embeddings.json"


# ---------------------------------------------------------------- 1. RAG
def load_document():
    """Find it_sector.txt next to this file, or in the repo's Docs folder."""
    candidates = [
        BASE_DIR / "it_sector.txt",
        BASE_DIR.parent / "Docs" / "Assessments" / "Day-12" / "it_sector.txt",
        BASE_DIR.parent.parent / "Docs" / "Assessments" / "Day-12" / "it_sector.txt",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("it_sector.txt not found. Copy it next to app_implementation.py")


def chunk_by_section(text):
    """Split the FAQ into one chunk per '## ' section."""
    chunks = []
    for part in text.split("\n## "):
        part = part.replace("---", "").strip()
        if part:
            chunks.append(part if part.startswith("##") else "## " + part)
    return chunks


def generate_embedding(texts, input_type="document"):
    result = voyage_client.embed(texts, model=embedding_model, input_type=input_type)
    return result.embeddings


def embed_documents_cached(chunks):
    """Embed the FAQ once and save to faq_embeddings.json.
    Later runs reuse the file, so only user questions cost Voyage tokens.
    The cache is rebuilt automatically if the FAQ text or model changes."""
    key = hashlib.sha256((embedding_model + "".join(chunks)).encode()).hexdigest()
    if EMBEDDING_CACHE.exists():
        cached = json.loads(EMBEDDING_CACHE.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached["embeddings"]
    embeddings = generate_embedding(chunks, input_type="document")
    EMBEDDING_CACHE.write_text(json.dumps({"key": key, "embeddings": embeddings}), encoding="utf-8")
    return embeddings


class VectorIndex:
    """Very small in-memory vector database using cosine similarity."""

    def __init__(self):
        self.vectors = []
        self.documents = []

    def add_vector(self, vector, document):
        self.vectors.append(vector)
        self.documents.append(document)

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def search(self, query_vector, k=2):
        scored = [(self._cosine(query_vector, v), d) for v, d in zip(self.vectors, self.documents)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def build_index():
    chunks = chunk_by_section(load_document())
    embeddings = embed_documents_cached(chunks)
    index = VectorIndex()
    for embedding, chunk in zip(embeddings, chunks):
        index.add_vector(embedding, chunk)
    return index


def retrieve(index, query, k=2):
    query_vector = generate_embedding([query], input_type="query")[0]
    return [doc for score, doc in index.search(query_vector, k)]


# ---------------------------------------------------------------- 2. Tools
web_search_tool = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3,
}

datetime_tool = {
    "name": "get_current_datetime",
    "description": "Returns the user's current local date and time.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def get_current_datetime():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


# ---------------------------------------------------------------- 3. MCP client
async def _list_mcp_tools():
    async with Client(MCP_SERVER_URL) as mcp_client:
        result = await mcp_client.list_tools()
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in result.tools
        ]


async def _call_mcp_tool(name, arguments):
    async with Client(MCP_SERVER_URL) as mcp_client:
        result = await mcp_client.call_tool(name, arguments)
        return "\n".join(c.text for c in result.content if c.type == "text")


def get_mcp_tools():
    return asyncio.run(_list_mcp_tools())


def run_tool(name, tool_input):
    if name == "get_current_datetime":
        return get_current_datetime()
    return asyncio.run(_call_mcp_tool(name, tool_input))  # MCP tools


# ---------------------------------------------------------------- Chat
SYSTEM_PROMPT = """You are an IT Helpdesk Assistant for company employees.
- Each user message starts with <it_faq_context>: FAQ sections retrieved for that
  question. Use them when they answer the question; ignore them when they don't.
- Use web_search when the FAQ does not cover the question or the user asks to search.
- Use create_support_ticket when the user wants to raise a ticket (ask for the employee's name if unknown).
- Use check_ticket_status when the user asks about a ticket ID like IT-101.
- Use get_current_datetime for questions about the current date or time. It returns
  the user's local date and time; report it as-is without comparing to any other date.
Give clear, step-by-step troubleshooting answers. Keep responses concise."""


def chat(messages, user_text, index, mcp_tools):
    """Run one turn. Returns a dict with answer, tool_calls, faq_sections, sources."""
    # 1. RAG: always retrieve, and keep the context inside this turn's user message
    retrieved = retrieve(index, user_text)
    context = "\n\n".join(retrieved)
    messages.append({
        "role": "user",
        "content": f"<it_faq_context>\n{context}\n</it_faq_context>\n\nQuestion: {user_text}",
    })

    tools = [web_search_tool, datetime_tool, *mcp_tools]
    tool_calls = []  # {name, input, output} for the UI
    sources = []     # {title, url} from web search

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": response.content})

        # Server-side web search: record the query and result links
        for block in response.content:
            if block.type == "server_tool_use":
                tool_calls.append({"name": block.name, "input": block.input,
                                   "output": "Run by Anthropic (web search)"})
            elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                for r in block.content:
                    if getattr(r, "url", None) and r.url not in [s["url"] for s in sources]:
                        sources.append({"title": getattr(r, "title", r.url), "url": r.url})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    output, is_error = run_tool(block.name, block.input), False
                except Exception as e:
                    output, is_error = f"Error: {e}", True
                tool_calls.append({"name": block.name, "input": block.input, "output": output})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "pause_turn":
            continue  # long web search: let Claude continue
        else:
            break

    return {
        "answer": "".join(b.text for b in response.content if b.type == "text"),
        "tool_calls": tool_calls,
        "faq_sections": [doc.splitlines()[0].lstrip("# ") for doc in retrieved],
        "sources": sources,
    }
