"""
One-time ingestion script that built this repo's chroma_db/ vector store.

Loads John Keells Holdings' Annual Report 2025/26 (pages 0-60 and 138-160,
core narrative and outlook), chunks it with LangChain's
RecursiveCharacterTextSplitter (500 characters, 50 overlap, chosen
empirically), embeds each chunk via the Gemini embedding API, and stores
the result in ChromaDB. app.py serves queries against the resulting store;
it does not re-run this script.

Requires the source PDF locally as "Annual Report.pdf" (not committed to
this repo due to its size) and GEMINI_API_KEY set in the environment.
Resumable: re-running after a partial failure picks up from the last
successfully stored batch.
"""

import time
from google import genai
import chromadb
from chromadb import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

client_genai = genai.Client()

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
            print(f"    Error: {type(e).__name__} - {e}")
            print(f"    Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed after {max_retries} attempts.")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        embeddings = []
        for text in input:
            embeddings.append(get_embedding(text))
            time.sleep(0.7)
        return embeddings

db_client = chromadb.PersistentClient(path="./chroma_db")

collection = db_client.get_or_create_collection(
    name="day10_documents",
    embedding_function=GeminiEmbeddingFunction()
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def load_pdf(path, start_page=0, end_page=None):
    reader = PdfReader(path)
    end_page = end_page or len(reader.pages)
    full_text = ""
    for page in reader.pages[start_page:end_page]:
        text = page.extract_text()
        if text:
            full_text += text + "\n\n"
    return full_text

def chunk_text(text):
    return splitter.split_text(text)


def ingest(path, batch_size=50):
    print("Loading PDF (two sections: core narrative + outlook/strategy)...")
    text_core = load_pdf(path, start_page=0, end_page=60)
    text_outlook = load_pdf(path, start_page=138, end_page=160)
    text = text_core + text_outlook
    print(f"Loaded {len(text)} characters.")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    already_stored = collection.count()
    print(f"Collection already has {already_stored} chunks stored (resuming from here).")

    total = len(chunks)
    for batch_start in range(already_stored, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = chunks[batch_start:batch_end]
        batch_ids = [f"chunk_{i}" for i in range(batch_start, batch_end)]

        try:
            collection.add(documents=batch_chunks, ids=batch_ids)
            print(f"Saved batch: chunks {batch_start}-{batch_end - 1} ({collection.count()}/{total} total)")
        except Exception as e:
            print(f"BATCH FAILED at chunks {batch_start}-{batch_end - 1}: {e}")
            print("Stopping here. Already-saved chunks are safe. Re-run the script to resume from this point.")
            return

    print("Ingestion complete.")


ingest("Annual Report.pdf")
print("Final documents in collection:", collection.count())


results = collection.query(
    query_texts=["What are the biggest risks facing the company?"],
    n_results=3
)
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc[:150]}...")