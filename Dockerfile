FROM python:3.11-slim

WORKDIR /app

# Run as root: bind mounts for ./data and "./research reviews" avoid uid/gid
# mismatches on Windows and macOS hosts. For a production image with no host
# volumes, add a non-root USER after chown on /app/data and "/app/research reviews".

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY main.py streamlit_app.py ./
COPY config/ config/
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY eval/ eval/
COPY tests/ tests/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    DATABASE_PATH=/app/data/papers.sqlite3

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
