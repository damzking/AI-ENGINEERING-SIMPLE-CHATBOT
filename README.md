# AI Engineering Simple Chatbot

This repository contains an early Retrieval Augmented Generation (RAG) chatbot experiment built in a Jupyter/Google Colab notebook. The current focus is PDF-based question answering: a user uploads a PDF, the document is split into smaller text chunks, the chunks are embedded into a vector store, and a language model uses the retrieved context to answer questions from the document.

The main project file is:

```text
PYPDF.ipynb
```

## Project Goal

The goal of this project is to explore how Retrieval Augmented Generation can make chatbot answers more grounded and document-aware. Instead of relying only on a model's general knowledge, the chatbot retrieves relevant text from an uploaded PDF before generating an answer.

This is useful for working with:

- academic papers,
- admission or application documents,
- reports,
- letters,
- research notes,
- long PDFs that are difficult to inspect manually.

## What the Notebook Does

The notebook currently demonstrates the core stages of a PDF-based RAG pipeline:

1. Upload a PDF document in Google Colab.
2. Load the PDF with LangChain's `PyPDFLoader`.
3. Inspect the extracted page content.
4. Split the document into smaller overlapping chunks.
5. Create embeddings for the chunks using OpenAI embeddings.
6. Store the embedded chunks in a Chroma vector store.
7. Convert the vector store into a retriever.
8. Build a prompt that instructs the model to answer using only retrieved context.
9. Ask questions about the PDF through a simple LangChain RAG chain.

## Technologies Used

- Python
- Google Colab
- LangChain
- PyPDF
- OpenAI embeddings
- Chroma vector store
- OpenAI chat model

## Current Status

This repository is still in an exploratory notebook stage. The code is intentionally kept in `PYPDF.ipynb` while the RAG workflow is being developed and tested.

The current implementation is not yet a full production application. It does not currently include:

- a web interface,
- persistent vector database storage,
- automated tests,
- multi-PDF support,
- a packaged Python module,
- deployment configuration.

## Why This Project Matters

RAG is important because it helps reduce hallucination in language model applications. By retrieving source text before generating an answer, the system can produce responses that are more closely tied to the provided document.

For research and academic workflows, this approach can support faster document review and more transparent question answering over long-form text.

## Planned Improvements

Possible next steps include:

- cleaning the notebook into a more structured research demonstration,
- adding source citations for retrieved PDF chunks,
- saving and reusing the Chroma vector store,
- supporting multiple uploaded PDFs,
- creating a simple Streamlit or Gradio interface,
- adding an evaluation set of questions and expected answers,
- turning the notebook workflow into reusable Python scripts.

## How to Run

Open `PYPDF.ipynb` in Google Colab and run the cells from top to bottom.

You will need an OpenAI API key for the embedding and chat model steps.

The notebook installs the main dependencies inside Colab.
