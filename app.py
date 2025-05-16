import gradio as gr
import os
import io
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()

class CSVChatBot:
    def __init__(self):
        self.agent = None
        self.file_path = None

    def initialize_agent(self, file, api_key):
        if file is None or api_key.strip() == "":
            return "⚠ Please enter your API key and upload a CSV file."

        self.file_path = file.name
        try:
            model = ChatOpenAI(
                temperature=0.5,
                model="mistralai/mistral-7b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key  # 🔐 Use directly from input
            )

            self.agent = create_csv_agent(
                llm=model,
                path=self.file_path,
                allow_dangerous_code=True,  # ✅ Required for Python execution
                agent_executor_kwargs=dict(handle_parsing_errors=True)
            )
            return "🟢 CSV file uploaded and agent initialized successfully!"
        except Exception as e:
            return f"Oops!! Error occurred: {e}"

    def chat(self, message, history):
        if self.agent is None:
            return "Please initialize the agent first by uploading a CSV file and providing an API key."

        try:
            CSV_PROMPT_PREFIX = "First get the column names from the CSV file, then answer the question."
            CSV_PROMPT_SUFFIX = """
            - **ALWAYS** before giving the Final Answer, try another method.
            Then reflect on the answers of the two methods you did and ask yourself
            if it answers correctly the original question.
            If you are not sure, try another method.
            - If the methods tried do not give the same result, reflect and
            try again until you have two methods that have the same result.
            - If you still cannot arrive to a consistent result, say that
            you are not sure of the answer.
            - If you are sure of the correct answer, create a beautiful
            and thorough response using Markdown.
            - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
            ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
            - **ALWAYS**, as part of your "Final Answer", explain how you got
            to the answer on a section that starts with: "\n\nExplanation:\n".
            In the explanation, mention the column names that you used to get
            to the final answer.
            """

            result = self.agent.run(CSV_PROMPT_PREFIX + message + CSV_PROMPT_SUFFIX)

            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode()
                img_markdown = f"![plot](data:image/png;base64,{img_str})"
                plt.clf()
                plt.close(fig)
                return result + "\n\n" + img_markdown
            else:
                return result
        except Exception as e:
            return f"An error occurred: {str(e)}"

# UI Setup
csv_chatbot = CSVChatBot()

with gr.Blocks() as demo:
    gr.Markdown("# 📊 CSV Analysis Chatbot using OpenRouter AI")

    with gr.Row():
        file_input = gr.File(label="Upload CSV File")
        api_key_input = gr.Textbox(label="Enter OpenRouter API Key", type="password")

    initialize_button = gr.Button("Initialize Agent")
    init_output = gr.Textbox(label="Initialization Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question about the CSV", placeholder="e.g., What is the average sales?", lines=1)
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = csv_chatbot.chat(user_message, history)
        history[-1][1] = bot_message
        return history

    def clear_chat():
        return []

    initialize_button.click(csv_chatbot.initialize_agent, inputs=[file_input, api_key_input], outputs=init_output)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(clear_chat, None, chatbot, queue=False)

demo.launch()
