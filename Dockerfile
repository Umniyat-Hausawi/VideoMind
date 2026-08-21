FROM python:3.11-slim

# FFmpeg is required by input_layer.py for audio/frame extraction.
# git is required by yt-dlp for some extraction backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer) so Docker can cache
# this step and skip reinstalling on every code-only change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the five pipeline layers (as the "layers" package), the
# observability module app.py imports at startup, and the Streamlit
# entrypoint itself.
COPY layers/ ./layers/
COPY observability.py .
COPY app.py .

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
EXPOSE 7860

# --server.address 0.0.0.0 is required — Streamlit's default (localhost)
# is not reachable from outside the container.
# --server.headless true avoids Streamlit's interactive "email prompt"
# blocking container startup.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]