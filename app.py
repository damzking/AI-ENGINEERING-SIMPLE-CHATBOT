import hashlib
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import dotenv_values
from pypdf import PdfReader

ENV_SETTINGS = dotenv_values(Path(__file__).with_name(".env"))


def get_setting(name: str) -> str:
    return str(ENV_SETTINGS.get(name) or "").strip()


def pdf_to_chunks(pdf_bytes: bytes, filename: str) -> list[Document]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": filename, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    pages = [page for page in pages if page.page_content.strip()]
    if not pages:
        raise ValueError("No readable text was found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(pages)


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {document.metadata['source']}, page {document.metadata['page']}]\n"
        f"{document.page_content}"
        for document in documents
    )


def answer_question(
    question: str,
    vectorstore: Chroma,
    webhook_url: str,
    webhook_token: str,
) -> tuple[str, list[dict[str, str | int]]]:
    documents = vectorstore.similarity_search(question, k=4)
    if not documents:
        return "I couldn't find relevant information in this PDF.", []

    payload = {
        "question": question,
        "context": format_context(documents),
        "sources": [
            {"filename": document.metadata["source"], "page": document.metadata["page"]}
            for document in documents
        ],
    }
    response = requests.post(
        webhook_url,
        json=payload,
        headers={"Authorization": "Bearer " + webhook_token},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("The n8n webhook returned an unexpected response.")
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The n8n workflow returned no answer.")

    sources: list[dict[str, str | int]] = []
    seen_sources: set[tuple[str, int]] = set()
    response_sources = result.get("sources", payload["sources"])
    if not isinstance(response_sources, list):
        raise RuntimeError("The n8n workflow returned invalid source references.")
    for source_item in response_sources:
        if (
            not isinstance(source_item, dict)
            or not isinstance(source_item.get("filename"), str)
            or not isinstance(source_item.get("page"), int)
        ):
            raise RuntimeError("The n8n workflow returned an invalid source reference.")
        source = (source_item["filename"], source_item["page"])
        if source not in seen_sources:
            sources.append({"filename": source[0], "page": source[1]})
            seen_sources.add(source)

    return answer.strip(), sources


st.set_page_config(page_title="PDF Chat")
st.title("Chat with a PDF")
st.write("Upload a PDF and ask questions grounded in its contents.")

with st.sidebar:
    st.header("Settings")
    embedding_model = st.text_input(
        "Local embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="Downloaded and run locally; it does not require an inference API call.",
    ).strip()
    st.warning(
        "n8n runs Gemini through Gateway credits. Each model call uses your "
        "instance's shared credit balance."
    )

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

webhook_url = get_setting("N8N_WEBHOOK_URL")
webhook_token = get_setting("N8N_WEBHOOK_TOKEN")

if not webhook_url or not webhook_token:
    st.info(
        "Set N8N_WEBHOOK_URL and N8N_WEBHOOK_TOKEN in the project's .env file "
        "before asking questions."
    )

if "index_id" not in st.session_state:
    st.session_state.index_id = None
    st.session_state.vectorstore = None
    st.session_state.messages = []

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    content_id = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = hashlib.sha256(
        f"{content_id}:{uploaded_file.name}:{embedding_model}".encode()
    ).hexdigest()

    if webhook_url and webhook_token:
        if st.session_state.index_id != document_id:
            with st.spinner("Reading the PDF and creating its local search index..."):
                chunks = pdf_to_chunks(pdf_bytes, uploaded_file.name)
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )
                st.session_state.index_id = document_id
                st.session_state.messages = []

        if st.session_state.index_id == document_id:
            st.success(f"Ready to chat about **{uploaded_file.name}**.")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        st.caption(
                            "Sources: "
                            + "; ".join(
                                f"{source['filename']}, page {source['page']}"
                                for source in message["sources"]
                            )
                        )

            if question := st.chat_input("Ask a question about this PDF"):
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching the PDF and sending it to n8n..."):
                        try:
                            answer, sources = answer_question(
                                question,
                                st.session_state.vectorstore,
                                webhook_url,
                                webhook_token,
                            )
                        except requests.RequestException as error:
                            st.error(
                                "Could not reach the n8n workflow. Check that the "
                                "workflow is active, the production URL is correct, "
                                "and the webhook Header Auth credential matches. "
                                f"Details: {error}"
                            )
                        except (ValueError, RuntimeError) as error:
                            st.error(f"The n8n workflow response was invalid: {error}")
                        else:
                            st.markdown(answer)
                            if sources:
                                st.caption(
                                    "Sources: "
                                    + "; ".join(
                                        f"{source['filename']}, page {source['page']}"
                                        for source in sources
                                    )
                                )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": sources,
                                }
                            )
