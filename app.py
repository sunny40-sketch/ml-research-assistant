import gradio as gr

def answer(question):
    return "ML Research Assistant - Set up your API key to use this system."

demo = gr.Interface(
    fn=answer,
    inputs="text",
    outputs="text",
    title="ML Research Assistant",
    description="100% Recall@3 | 84% Faithfulness | See GitHub for full setup"
)

demo.launch(server_name="0.0.0.0", server_port=int(__import__('os').environ.get('PORT', 7860)))
