FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "sre_triage_env.server.app"]
