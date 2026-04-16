import os

from dotenv import load_dotenv
from nicegui import ui

from agent import GermanTeacherAgent
from chat_ui import ChatUI

load_dotenv()


def main():
    api_key = os.environ["GEMINI_API_KEY"]
    agent = GermanTeacherAgent(api_key)
    chat = ChatUI(agent)
    chat.build()
    ui.run(title="Deutsch Lehrer")


main()
