# =============================================================================
# Aegis-Router: Standalone Development Dockerfile
# Base: Python 3.13 (matching swe-arena-base reference)
# Target: Full dev environment for cache-aware LLM orchestrator
# =============================================================================

FROM python:3.13-slim-bookworm

# ---------------------------------------------------------------------------
# 1. System Dependencies
#    - Build tools for compiling llama-cpp-python from source (C++ backend)
#    - Shared memory support (POSIX IPC via kernel)
#    - Debugging & profiling tools for development inside the container
#    - Process management for running 5 model pods
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # === Build toolchain (llama.cpp compilation) ===
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    pkg-config \
    # === SSL & crypto (for pip, HTTPS downloads) ===
    libssl-dev \
    libffi-dev \
    # === Python dev headers (C extension compilation) ===
    python3-dev \
    # === Network tools ===
    wget \
    curl \
    netcat-openbsd \
    iproute2 \
    dnsutils \
    # === Version control ===
    git \
    # === Terminal multiplexer (manage 5 pods + router simultaneously) ===
    tmux \
    screen \
    # === System monitoring & debugging ===
    htop \
    iotop \
    strace \
    ltrace \
    gdb \
    # === Process management ===
    supervisor \
    # === Shared memory tools (inspect /dev/shm, ipcs for IPC debugging) ===
    util-linux \
    # === Text editors for in-container development ===
    vim \
    nano \
    # === Misc utilities ===
    less \
    jq \
    tree \
    procps \
    lsof \
    bash-completion \
    # === Graphviz (for rendering Trie/architecture diagrams) ===
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Configure shared memory limits
#    Default Docker /dev/shm is 64MB — far too small for our 10GB KV-cache pool.
#    This can't be set in Dockerfile (it's a runtime flag), but we document it
#    and create the config for supervisord.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Python Dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install build tools first, then install all requirements.
# CMAKE_ARGS ensures llama-cpp-python compiles with OpenBLAS for faster CPU inference.
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Pre-download the TinyLlama model (offline capability)
#    TinyLlama-1.1B Q4_K_M: ~670MB, optimal for 5 pods on 6 CPUs / 40GB RAM
# ---------------------------------------------------------------------------
RUN mkdir -p /models && \
    wget -q --show-progress \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -O /models/tinyllama.gguf

# ---------------------------------------------------------------------------
# 5. Supervisord config for managing 5 model pods
# ---------------------------------------------------------------------------
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor

COPY configs/supervisord.conf /etc/supervisor/conf.d/aegis-router.conf

# ---------------------------------------------------------------------------
# 6. Create project skeleton directories
# ---------------------------------------------------------------------------
RUN mkdir -p /app/router /app/workers /app/shared /app/tests /app/benchmarks /app/configs

# ---------------------------------------------------------------------------
# 7. Copy project files
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# 8. Environment Variables
# ---------------------------------------------------------------------------
# Inference engine
ENV LLAMA_CPP_LIB=/usr/local/lib/python3.13/site-packages/llama_cpp/libllama.so
ENV GGML_USE_OPENBLAS=1

# CPU threading (6 CPUs across 5 pods = ~1.2 threads per pod)
ENV OMP_NUM_THREADS=1
ENV LLAMA_N_THREADS=1

# Shared memory config
ENV AEGIS_SHM_POOL_SIZE=10737418240
ENV AEGIS_SHM_BLOCK_SIZE=8388608
ENV AEGIS_MODEL_PATH=/models/tinyllama.gguf
ENV AEGIS_NUM_PODS=5
ENV AEGIS_MAX_CPU_LOAD=90

# Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Misc
ENV TERM=xterm-256color
ENV SHELL=/bin/bash

# ---------------------------------------------------------------------------
# 9. Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["/bin/bash"]
