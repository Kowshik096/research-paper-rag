import json
import os

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def chat(question: str, history: list, top_k: int):
    if not question.strip():
        yield history, "Ask a question first."
        return

    history = history + [[question, ""]]
    sources_md = ""
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/api/v1/ask/stream",
            json={"question": question, "top_k": int(top_k)},
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "sources":
                    rows = [
                        f"**[{i}] {s['title']}** — *{s['section']}*"
                        for i, s in enumerate(event["data"], 1)
                    ]
                    sources_md = "\n\n".join(rows) if rows else "_No sources found._"
                    yield history, sources_md
                elif event["type"] == "token":
                    history[-1][1] += event["data"]
                    yield history, sources_md
                elif event["type"] == "error":
                    history[-1][1] = event["data"]
                    yield history, sources_md
    except httpx.HTTPError as exc:
        history[-1][1] = f"API error: {exc}. Is the backend running on {API_URL}?"
    yield history, sources_md


with gr.Blocks(title="Hybrid RAG Research Assistant") as demo:
    gr.Markdown("# Hybrid RAG Research Assistant\nAnswers grounded in indexed arXiv papers, with citations.")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480)
            question = gr.Textbox(
                placeholder="e.g. How do mesoscale eddies affect ocean heat transport?", label="Question"
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.ClearButton([chatbot, question])
            top_k = gr.Slider(2, 12, value=6, step=1, label="Chunks to retrieve (top-k)")
        with gr.Column(scale=1):
            sources = gr.Markdown("### Sources\n_Citations appear here after you ask._")

    ask_btn.click(chat, [question, chatbot, top_k], [chatbot, sources])
    question.submit(chat, [question, chatbot, top_k], [chatbot, sources])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
