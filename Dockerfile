FROM runpod/base:0.6.3-cpu

# Set python3.10 as default
RUN ln -sf $(which python3.10) /usr/local/bin/python && \
    ln -sf $(which python3.10) /usr/local/bin/python3

# Upgrade pip & install uv early to avoid caching issues
RUN pip install --upgrade pip && pip install uv

# Copy requirements and install Python packages using uv (cleaner env)
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --no-cache-dir --system

COPY handler.py ./

# Entrypoint
CMD python -u /handler.py
