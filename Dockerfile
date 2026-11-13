FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install .

COPY src/ src/
COPY configs/ configs/
COPY contracts/ contracts/

EXPOSE 8000 8501

CMD ["uvicorn", "trustgate.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
