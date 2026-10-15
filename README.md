# AI Engineering Simple Chatbot

This repository contains a PDF-based Retrieval Augmented Generation (RAG) chatbot. It includes both an exploratory Jupyter/Google Colab notebook and a simple Streamlit application for chatting with an uploaded PDF.

The main files are:

```text
app.py
PYPDF.ipynb
```

## n8n Cloud + Streamlit

The Streamlit app reads the PDF and retrieves relevant text locally. It sends the question and retrieved excerpts to an n8n Cloud webhook. An n8n AI Agent uses the **Google Gemini Chat Model** (default: Gemini 2.5 Flash) and Calculator tool, then returns the answer and page sources. The Gemini model runs through n8n **Gateway credits**, so no Hugging Face or Google API credential is needed. Locally, webhook settings come from `.env`; on Streamlit Community Cloud, they come from the app's protected Secrets settings. Local `.env` values take precedence.

### Set up the n8n workflow

1. In n8n Cloud, import [`n8n/pdf_chat_workflow.json`](./n8n/pdf_chat_workflow.json).
2. Create a **Header Auth** credential for the Webhook node. Set the header name to `Authorization` and the value to `Bearer <your webhook secret>`. Generate a secret locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"`; use the same secret for `N8N_WEBHOOK_TOKEN` in Streamlit, without the `Bearer ` prefix.
3. On the **Gemini Chat Model** node, select **Use Gateway credits** in the credential field, and choose `models/gemini-2.5-flash` if available. The imported workflow includes a Calculator tool so the AI Agent can handle arithmetic questions grounded in the PDF. Gateway credits are currently available on n8n Cloud Starter and Pro plans (including free trials), but not Enterprise; the instance owner can also disable them.
4. Select the Webhook Header Auth credential, save and activate/publish the workflow, then copy its **Production URL** from the Webhook node.

Gateway credits are usage-based and shared across the n8n Cloud instance. n8n says new Cloud users and free trials receive a small amount of free Gateway credits; check your balance and model rates in n8n before testing. Requests send the question and retrieved PDF excerpts to Gemini through n8n's gateway.

### Run locally

Copy `.env.example` to `.env`, and set `N8N_WEBHOOK_URL` to the production URL and `N8N_WEBHOOK_TOKEN` to the webhook secret you created. Install the dependencies and run the app:

```powershell
Copy-Item .env.example .env
# Edit .env with your production webhook URL and webhook secret.
pip install -r requirements.txt
streamlit run app.py
```

The app builds an in-memory Chroma index with `sentence-transformers/all-MiniLM-L6-v2` and displays source pages. The index and chat history are temporary and are cleared when the Streamlit session ends or a different PDF or embedding model is selected.

### Deploy on Streamlit Community Cloud

1. Push the project to GitHub. The deployment branch for this version is `deploy/pdf-chat`.
2. In Streamlit Community Cloud, create an app from `damzking/AI-ENGINEERING-SIMPLE-CHATBOT`, select branch `deploy/pdf-chat`, and set the main file to `app.py`.
3. In **Advanced settings → Secrets**, add the n8n production URL and the Header Auth secret. Do not include `Bearer ` in the token value:

   ```toml
   N8N_WEBHOOK_URL = "https://emmanuel201.app.n8n.cloud/webhook/pdf-chat"
   N8N_WEBHOOK_TOKEN = "your-webhook-secret-without-Bearer-prefix"
   ```

4. Save and deploy. Keep the real `.env` file out of Git; `.gitignore` excludes it. For local runs, use `.env` as described above.

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
- Streamlit
- n8n Cloud webhooks
- Google Colab
- LangChain
- PyPDF
- Chroma vector store
- Local Sentence Transformers embeddings
- OpenAI embeddings and chat model in the exploratory notebook

## Current Status

The notebook remains an exploratory record of the RAG workflow. The Streamlit app and importable n8n workflow provide a basic local/cloud setup, but the project is not yet production-ready and does not currently include:

- persistent vector database storage,
- automated tests,
- multi-PDF support,
- a packaged Python module,
- automated deployment.

## Why This Project Matters

RAG is important because it helps reduce hallucination in language model applications. By retrieving source text before generating an answer, the system can produce responses that are more closely tied to the provided document.

For research and academic workflows, this approach can support faster document review and more transparent question answering over long-form text. The app provides a practical n8n webhook integration and a local embedding model.

## Planned Improvements

Possible next steps include:

- cleaning the notebook into a more structured research demonstration,
- improving citations with quoted excerpts from retrieved PDF chunks,
- saving and reusing the Chroma vector store,
- supporting multiple uploaded PDFs,
- adding the planned Tavily/Wikipedia question router,
- adding an evaluation set of questions and expected answers,
- turning the notebook workflow into reusable Python scripts.

## How to Run

For the interactive app, follow the [n8n Cloud and Streamlit instructions above](#n8n-cloud--streamlit). To explore the original notebook, open `PYPDF.ipynb` in Google Colab and run the cells from top to bottom; the notebook's model setup remains separate from the Streamlit app.
