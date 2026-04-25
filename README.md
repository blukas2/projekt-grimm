# Projekt Grimm

Projekt Grimm is a small local chat application for practicing German with a Gemini-powered teacher and a Gemma-powered translator. It combines Google GenAI chat and translation components with a NiceGUI interface, so you can open the app in your browser and work with a teacher persona alongside a dedicated translator panel.

## What It Does

- Responds as a German teacher for B1/B2-level learners.
- Provides a separate English-German translator panel powered by Gemma 3 27B.
- Expects the student to write in German in normal conversation.
- Allows exceptions for translation requests, grammar explanations, and exercise instructions.
- Grades completed exercises on a `1` to `5` scale.
- Enriches single-word translations with German noun and verb metadata.
- Shows the lesson chat and translator in a lightweight browser-based UI.

## Tech Stack

- Python
- `google-genai`
- `nicegui`
- `python-dotenv`

## Requirements

- A recent Python 3 version
- A Google Gemini API key

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the repository root.

```env
GEMINI_API_KEY=your_api_key_here
```

## Run The App

Start the application with:

```powershell
python main.py
```

NiceGUI will start a local web server and print the local address in the terminal. In a default setup this is usually `http://localhost:8080`.

## Project Structure

```text
app/
	core/
		logging.py          Central application logger with five-minute retention
	agents/
		german_teacher.py   Gemini-backed German teacher agent and system prompt
		translator.py       Gemma-backed translator with lexical result normalization
	ui/
		chat.py             NiceGUI lesson chat plus translator panel
main.py                 Application entrypoint
```

## How It Works

- `main.py` loads environment variables, creates the teacher and translator components, builds the UI, and starts NiceGUI.
- `app/core/logging.py` writes application logs to `.logs/application.log` and keeps only the last five minutes of entries.
- `app/agents/german_teacher.py` configures the Gemini chat session and defines the teacher behavior through a system prompt.
- `app/agents/translator.py` builds Gemma translation prompts, classifies single-word lookups, and normalizes lexical results.
- `app/ui/chat.py` renders the lesson chat, the translator panel, and their separate interactions.

## Notes

- The app expects `GEMINI_API_KEY` to be present; otherwise startup will fail.
- The app creates `.logs/application.log` automatically and prunes entries older than five minutes whenever a new log is written.
- The current interface is intentionally minimal and optimized for local use.
