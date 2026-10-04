# Production image for the Streamlit app.
#
#   docker build -t jkh-rag .
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... jkh-rag
#
# Smaller image for low-memory hosts:
#   docker build --build-arg RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 -t jkh-rag-small .

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf

WORKDIR /app
RUN groupadd -r rag && useradd -r -g rag -d /app rag

# CPU-only torch first: the default Linux wheel pulls ~2 GB of CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the reranker into the image so containers start without downloading
# 1 GB from Hugging Face, and can run with no outbound access to it.
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV RAG_MODELS_RERANKER=${RERANKER_MODEL}
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}')"
ENV HF_HUB_OFFLINE=1

# App code, config and the prebuilt vector index (chroma_db/).
COPY --chown=rag:rag . .
RUN mkdir -p /app/.cache && chown -R rag:rag /app

USER rag
EXPOSE 8501

# Real liveness check: Streamlit's own health endpoint must answer 200.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
