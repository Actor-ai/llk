Create `.env` file from `.env-example`,

Run:
```commandline
git clone git@github.com:Actor-ai/llk.git
docker build -t nvidia-gpu:latest .
docker compose up -d
# docker exec ollama_llm ollama pull orca2:13b
docker exec ollama_llm ollama pull mistral:7b-instruct
```
than go to `http://<your_ip>:8501`.