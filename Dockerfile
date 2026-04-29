FROM dustynv/l4t-pytorch:r36.4.0

WORKDIR /home/code

ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Copiamos el codigo del proyecto. En desarrollo se montara tambien como volumen.
COPY . /home/code

CMD ["python3", "/home/code/main.py"]
