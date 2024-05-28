<h1 align="center">
📖Chat With Documents
</h1>

**Accurate answers and instant citations for your documents.**

Upload your documents and get answers to your questions, with citations from the text.

[Demo](https://twitter.com/mm_sasmitha/status/1620999984085884930)

## Installation

Follow the instructions below to run the Streamlit server locally.

### Pre-requisites

Make sure you have Python ≥3.10 installed.

### Steps

1. Install dependencies with [Poetry](https://python-poetry.org/) and activate virtual environment

```bash
poetry install
poetry shell
```

3. Add `.env` file to top level.

See `.env.example` for how the file should look like.

4. Run the Streamlit server

```bash
cd app
streamlit run main.py
```

## Build with Docker

⚠️ **Under construction** ⚠️

Run the following commands to build and run the Docker image.

```bash
docker build -t app .
docker run -p 8501:8501 app
```

Open http://localhost:8501 in your browser to access the app.

## Customization

You can increase the max upload file size by changing `maxUploadSize` in `.streamlit/config.toml`.
Currently, the max upload size is 25MB for the hosted version.
