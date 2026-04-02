FROM nvcr.io/nvidia/pytorch:24.12-py3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      nano curl ffmpeg \
      libsm6 libxext6 libnss3 \
      libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 \
      libasound2t64 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev libpango-1.0-0 \
      python3-lxml \
    && ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime \
    && echo "Europe/Zurich" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create python venv with uv and weasyprint installed
RUN python -m venv .venv \
 && .venv/bin/pip install --no-cache-dir uv weasyprint

# Install dependencies (cached unless uv.lock changes)
COPY pyproject.toml uv.lock /app/
RUN .venv/bin/uv sync --frozen --no-install-project --extra all,cu126

# Install mmore from local source code
COPY . /app
RUN .venv/bin/uv pip install --no-cache --no-deps -e .

# Runtime config
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT ["/bin/bash"]
