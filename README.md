<h1 align="center">
📖KnowledgeGPT
</h1>

**Accurate answers and instant citations for your documents.**

Upload your documents and get answers to your questions, with citations from the text.

## Installation

Follow the instructions below to run the Streamlit server locally.

### Pre-requisites

Make sure you have Python ≥3.10 installed.

### Steps

1. Clone the repository

```bash
git clone https://github.com/mmz-001/knowledge_gpt
cd knowledge_gpt
```

2. Install dependencies with [Poetry](https://python-poetry.org/) and activate virtual environment

```bash
poetry install
poetry shell
```

3. (Optional) Avoid adding the OpenAI API every time you run the server by adding it to environment variables.
   - Make a copy of `.env.example` named `.env`
   - Add your API key to the `.env` file

> **Note:** Make sure you have a paid OpenAI API key for faster completions and to avoid hitting rate limits.

4. Run the Streamlit server

```bash
cd knowledge_gpt
streamlit run main.py
```

## Build with Docker

Run the following commands to build and run the Docker image.

```bash
cd knowledge_gpt
docker build -t knowledge_gpt .
docker run -p 8501:8501 knowledge_gpt
docker container ls
docker stop adoring_mccarthy
```

Open http://localhost:8501 in your browser to access the app.

## Customization

You can increase the max upload file size by changing `maxUploadSize` in `.streamlit/config.toml`.
Currently, the max upload size is 25MB for the hosted version.

## Tech Stack

- User Interface - [Streamlit](https://streamlit.io/)
- LLM Tooling - [Langchain](https://github.com/hwchase17/langchain)

## Roadmap

- Add support for more formats (e.g. webpages, PPTX, etc.)
- Highlight relevant phrases in citations
- Support scanned documents with OCR
- More customization options (e.g. chain type, chunk size, etc.)
- Visual PDF viewer
- Support for Local LLMs
