# syntax=docker/dockerfile:1
# Streamlit UI image (HTTP client only; no ML libraries). Build from the repository root.
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install "streamlit" "httpx" "pyvis"

FROM python:3.11-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PATENTRAG_API_URL=http://api:8000
RUN useradd --create-home --uid 10001 patentrag
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=patentrag app /app/app
USER patentrag
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
