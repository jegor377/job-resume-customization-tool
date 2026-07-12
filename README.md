# Resume Customization Tool

An interactive, conversational assistant that helps you build a candidate profile once and then generate a **tailored job resume (PDF)** for any specific job offer — driven entirely through a chat interface.

Built with **LangGraph** (state machine + `interrupt()`-based human-in-the-loop flow), a **local LLM via Ollama**, and a **Streamlit** front end.

https://github.com/user-attachments/assets/7f8c26df-775a-42e1-bb20-74bd08e383fc

## How it works

1. **First run:** the assistant asks you to describe your commercial experience and skills in free text, extracts a structured `UserProfile` from it, and saves it locally.
2. **Main menu:** from there you can:
   - generate a resume for a specific job offer,
   - view your current profile,
   - edit your profile (in free-text form — the assistant figures out what changed).
3. **Job offer flow:** paste a job offer, the assistant extracts structured `JobRequirements` (must-have / nice-to-have skills), lets you confirm or correct them, then builds a `ResumeConfig` tailored to that specific offer by comparing your profile against the requirements.
4. **Generation:** you pick a resume language (based on available LaTeX templates), the assistant translates/tailors the config if needed, and generates a PDF resume.
5. **Review:** accept the generated resume, or ask for changes and regenerate.

The whole flow is implemented as a LangGraph `StateGraph` where each conversational step is a graph node, and turns are driven by `interrupt()` / `Command(resume=...)`, so the app can pause for user input at any point and resume exactly where it left off — including across Streamlit reruns, using a checkpointer keyed by a per-session `thread_id`.

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — state machine, human-in-the-loop interrupts, checkpointed conversation state
- **[LangChain](https://github.com/langchain-ai/langchain) + `langchain-ollama`** — structured output extraction (`with_structured_output`) from a local LLM
- **[Ollama](https://ollama.com/)** — local LLM inference
- **[Streamlit](https://streamlit.io/)** — chat UI and PDF preview
- **[Pydantic](https://docs.pydantic.dev/)** — schemas for `UserProfile`, `JobRequirements`, `ResumeConfig`, and internal graph state
- **LaTeX** (via `resume_generator`) — resume PDF rendering from language-specific templates

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally, with a suitable model pulled, e.g.:
  ```bash
  ollama pull llama3.2:3b
  ```
- A LaTeX distribution available on the system for PDF resume generation

## Installation

```bash
git clone <this-repo-url>
cd <repo-directory>
```

## Running

```bash
uv run streamlit run main.py
```

Then open the printed local URL in your browser and start chatting with the assistant.

## Project structure

```
.
├── app.py                     # LangGraph graph definition + Streamlit driver (this file)
├── schemas.py                 # Domain models: UserProfile, JobRequirements, Experience
├── resume_generator/
│   ├── generation.py          # generate_resume(): renders a ResumeConfig into a PDF
│   └── schemas.py             # ArtifactsConfig, ResumeConfig
├── user.json                  # Saved candidate profile (created on first run)
├── dump/                      # LaTeX templates for the job resumes. One per each desired language.
├── dump/                      # Generated tex build artifacts when something's failed (gitignored)
└── outputs/                   # Generated PDFs and intermediate build artifacts (gitignored)
```

## Configuration

- **LLM model / context window** — set in `app.py` when constructing `ChatOllama` (model name, `num_ctx`).
- **Templates & output paths** — controlled via `ArtifactsConfig` (`resume_generator/schemas.py`), including the templates directory used to detect available resume languages (`template_<lang>.tex`).
- **Profile storage** — currently a single local file, `user.json`, shared across sessions (single-user tool).

## Notes

- Conversation state (including your profile, the current job requirements, and the generated resume config) is checkpointed in memory per browser session (`InMemorySaver`), so it's **not persisted across app restarts** — only your `UserProfile` is persisted to `user.json`.
- The assistant relies on LLM-based intent classification for menu navigation and edit instructions; if it misclassifies your input, you can usually just re-state your intent more explicitly.
