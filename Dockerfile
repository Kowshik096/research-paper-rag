FROM python:3.11-slim

ENV HF_HOME=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache/hub
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache/hub

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
