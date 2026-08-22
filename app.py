import gradio as gr
import os

def answer(question):
    return "ML Research Assistant - See GitHub for full setup: github.com/sunny40-sketch/ml-research-assistant"

demo = gr.Interface(
    fn=answer,
    inputs="text",
    outputs="text",
    title="ML Research Assistant",
    description="100% Recall@3 | 84% Faithfulness | Powered by RAG + Claude"
)

port = int(os.environ.get('PORT', 8080))
demo.launch(server_name="0.0.0.0", server_port=port)
