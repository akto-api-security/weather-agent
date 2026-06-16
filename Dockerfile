FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py app.py tools.py session_store.py akto_middleware.py ./

EXPOSE 80

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
