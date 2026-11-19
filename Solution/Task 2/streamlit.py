
import json

import streamlit as st

from app_implementation import build_index, chat, get_mcp_tools

st.set_page_config(page_title="IT Support Chat Bot", page_icon="🛠️")
st.title("🛠️ IT Support Chat Bot")
st.caption("RAG on IT FAQ · Web search · Date/time tool · MCP tickets")


@st.cache_resource(show_spinner="Loading the IT FAQ vector index...")
def load_index():
    return build_index()


index = load_index()

# MCP tools (loaded once per session; app still works if the server is down)
if "mcp_tools" not in st.session_state:
    try:
        st.session_state.mcp_tools = get_mcp_tools()
    except Exception:
        st.session_state.mcp_tools = []

if "messages" not in st.session_state:
    st.session_state.messages = []  # full history sent to Claude
    st.session_state.history = []   # what the UI shows


def render_extras(result):
    """Show FAQ sections, tool calls (input/output) and web sources under a reply."""
    if result.get("faq_sections"):
        st.caption("📚 FAQ context: " + " · ".join(result["faq_sections"]))
    if result.get("tool_calls"):
        names = ", ".join(c["name"] for c in result["tool_calls"])
        with st.expander(f"🔧 Tools used: {names}"):
            for call in result["tool_calls"]:
                st.markdown(f"**{call['name']}**")
                st.markdown("Input:")
                st.code(json.dumps(call["input"], indent=2), language="json")
                st.markdown("Output:")
                st.code(call["output"])
    if result.get("sources"):
        links = " · ".join(f"[{s['title']}]({s['url']})" for s in result["sources"][:5])
        st.caption("🌐 Web sources: " + links)


with st.sidebar:
    st.header("Tools")
    st.write("✅ RAG: it_sector.txt")
    st.write("✅ web_search")
    st.write("✅ get_current_datetime")
    if st.session_state.mcp_tools:
        for tool in st.session_state.mcp_tools:
            st.write(f"✅ MCP: {tool['name']}")
    else:
        st.warning("MCP server not reachable. Run `python mcp_server.py` and click Reset.")

    st.header("Try")
    st.markdown(
        "- How do I fix network connectivity issues?\n"
        "- Search for the latest Windows 11 Wi-Fi troubleshooting steps.\n"
        "- Create a ticket for my laptop running slowly.\n"
        "- Check the status of ticket IT-101.\n"
        "- What is the current date and time?"
    )
    if st.button("Reset chat"):
        st.session_state.clear()
        st.rerun()

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["answer"])
        render_extras(item)

if prompt := st.chat_input("Ask an IT support question..."):
    st.session_state.history.append({"role": "user", "answer": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            saved = len(st.session_state.messages)
            try:
                result = chat(st.session_state.messages, prompt, index, st.session_state.mcp_tools)
            except Exception as e:
                del st.session_state.messages[saved:]  # drop the failed turn
                result = {"answer": f"⚠️ Error: {e}"}
        st.markdown(result["answer"] or "_(no text reply)_")
        render_extras(result)
    st.session_state.history.append({"role": "assistant", **result})
