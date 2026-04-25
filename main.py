import os

from dotenv import load_dotenv
from nicegui import ui

from app.agents import GermanTeacherAgent
from app.ui import ChatUI

load_dotenv()


def main():
    api_key = os.environ["GEMINI_API_KEY"]
    agent = GermanTeacherAgent(api_key)
    chat = ChatUI(agent)
    chat.build()
    ui.run(title="Deutsch Lehrer")


main()
