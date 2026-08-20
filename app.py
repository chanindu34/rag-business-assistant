import os
import time
from google import genai
import chromadb
from chromadb import EmbeddingFunction
import streamlit as st

# Works both locally (env var) and on Streamlit Cloud (st.secrets)
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
client_genai = genai.Client(api_key=api_key)

def get_embedding(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            result = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed after {max_retries} attempts.")

def generate_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client_genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError(f"Failed to generate after {max_retries} attempts.")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        embeddings = []
        for text in input:
            embeddings.append(get_embedding(text))
            time.sleep(0.7)
        return embeddings

# Path is now relative to THIS file's own folder — works identically
# whether run locally or on Streamlit Cloud, since the whole folder deploys together
db_client = chromadb.PersistentClient(path="chroma_db")

collection = db_client.get_or_create_collection(
    name="day10_documents",
    embedding_function=GeminiEmbeddingFunction()
)


def retrieve(query, k=6):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]


def build_prompt(query, chunks):
    numbered_context = ""
    for i, chunk in enumerate(chunks):
        numbered_context += f"[{i+1}] {chunk}\n\n"

    prompt = f"""Answer the question using ONLY the information in the context below.
Do NOT use any outside knowledge, even if you recognize the company or topic.
If specific details aren't in the context, explicitly say "the provided context doesn't cover this" rather than filling gaps from general knowledge.
Cite which source number(s) you used in brackets after each claim, like [1] or [1][3].

Context:
{numbered_context}

Question: {query}

Answer:"""
    return prompt


def answer(query, k=6):
    chunks = retrieve(query, k)
    prompt = build_prompt(query, chunks)
    answer_text = generate_with_retry(prompt)
    return {
        "answer": answer_text,
        "sources": chunks
    }


# --- Streamlit UI ---

st.title("Business Intelligence Assistant")
st.caption("Ask questions about John Keells Holdings' Annual Report 2025/26")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question about the report...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer(user_question)
                st.write(result["answer"])

                with st.expander("Sources"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**[{i+1}]** {source[:300]}...")

                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            except Exception as e:
                error_msg = "Sorry, I hit a temporary error reaching the AI service. Please try asking again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
