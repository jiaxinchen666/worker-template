FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set python3.10 as the default python
RUN ln -sf $(which python3.10) /usr/local/bin/python && \
    ln -sf $(which python3.10) /usr/local/bin/python3

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system
RUN apt-get update && apt-get install -y supervisor

COPY supervisord.conf /etc/supervisord.conf

# Add files
ADD handler.py .
ADD torch-flow.yml .
ADD img_prompt.py .
# Run the handler
CMD ["supervisord", "-c", "/etc/supervisord.conf"]