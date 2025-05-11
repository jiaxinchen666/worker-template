FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set python3.10 as default
RUN ln -sf $(which python3.10) /usr/local/bin/python && \
    ln -sf $(which python3.10) /usr/local/bin/python3

# Upgrade pip & install uv early to avoid caching issues
RUN pip install --upgrade pip && pip install uv

# Install system dependencies
RUN apt-get update && apt-get install -y supervisor

# Copy requirements and install Python packages using uv (cleaner env)
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --no-cache-dir --system

# Copy supervisor config and app code
COPY supervisord.conf /etc/supervisord.conf
COPY handler.py torch-flow.yml img_prompt.py ./

# Entrypoint
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
