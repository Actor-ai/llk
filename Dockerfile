FROM nvcr.io/nvidia/pytorch:23.11-py3
WORKDIR /app
COPY ./requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt
RUN curl https://ollama.ai/install.sh | sh
