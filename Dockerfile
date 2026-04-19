FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install tectonic (for PDF report generation)
RUN curl -fsSL \
    "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.0/tectonic-0.16.0-x86_64-unknown-linux-musl.tar.gz" \
    -o /tmp/tectonic.tar.gz \
    && tar -xz -C /usr/local/bin -f /tmp/tectonic.tar.gz \
    && rm /tmp/tectonic.tar.gz \
    && tectonic --version

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py agent.py rag.py knowledge_base.py ./

# Streamlit config
COPY .streamlit/config.toml /root/.streamlit/config.toml

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
