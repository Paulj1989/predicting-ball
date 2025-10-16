FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST
RUN useradd -m -u 1000 streamlit

# Copy packages to streamlit user's home directory
COPY --from=builder --chown=streamlit:streamlit /root/.local /home/streamlit/.local

# Copy application files
COPY --chown=streamlit:streamlit run.py ./
COPY --chown=streamlit:streamlit app/ ./app/
COPY --chown=streamlit:streamlit src/ ./src/
COPY --chown=streamlit:streamlit outputs/models/ ./outputs/models/
COPY --chown=streamlit:streamlit outputs/predictions/ ./outputs/predictions/
COPY --chown=streamlit:streamlit outputs/tables/ ./outputs/tables/
# COPY --chown=streamlit:streamlit data/ ./data/
COPY --chown=streamlit:streamlit .streamlit/ ./.streamlit/

# Switch to non-root user
USER streamlit

# Add user's local bin to PATH
ENV PATH=/home/streamlit/.local/bin:$PATH

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
