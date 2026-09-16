FROM python:3.12-slim

# China-friendly package mirrors. Override these build arguments when an
# official or private mirror is preferred.
ARG DEBIAN_MIRROR=mirrors.aliyun.com
ARG PYPI_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

ENV PIP_INDEX_URL=${PYPI_INDEX_URL} \
    UV_INDEX_URL=${PYPI_INDEX_URL}

WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install necessary dependencies, including FFmpeg, Calibre, and OpenGL
# libraries. python:slim may use either sources.list or debian.sources, so
# update both locations when present.
RUN set -eux; \
    for source_file in \
        /etc/apt/sources.list \
        /etc/apt/sources.list.d/debian.sources; do \
        if [ -f "${source_file}" ]; then \
            sed -i \
                -e "s|deb.debian.org|${DEBIAN_MIRROR}|g" \
                -e "s|security.debian.org|${DEBIAN_MIRROR}|g" \
                "${source_file}"; \
        fi; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    ffmpeg \
    curl \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    libnss3; \
    rm -rf /var/lib/apt/lists/*

# Install uv through the configured PyPI mirror
RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" uv

# Install Calibre. Its official installer has no stable official mainland
# mirror, so keep the trusted upstream URL and add retries for weak networks.
RUN curl -fsSL \
        --retry 5 \
        --retry-delay 3 \
        --connect-timeout 20 \
        https://download.calibre-ebook.com/linux-installer.sh \
        -o /tmp/calibre-installer.sh \
    && sh /tmp/calibre-installer.sh \
    && rm -f /tmp/calibre-installer.sh

# ✅ Make Calibre CLI tools globally accessible
RUN ln -s /opt/calibre/ebook-convert /usr/local/bin/ebook-convert && \
    ln -s /opt/calibre/ebook-meta /usr/local/bin/ebook-meta

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install Python dependencies (after upgrading pip)
RUN uv pip install \
    --system \
    --no-cache-dir \
    --no-deps \
    --index-url "${PYPI_INDEX_URL}" \
    -r requirements.txt

# Copy the rest of the application files
COPY . .

# ✅ Create the `generated_audiobooks` directory
RUN mkdir -p /app/generated_audiobooks

# Expose the port
EXPOSE 7860

# Run the application with Uvicorn
CMD ["uvicorn", "--access-log", "app:app", "--host", "0.0.0.0", "--port", "7860"]
