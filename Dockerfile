FROM --platform=$BUILDPLATFORM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV PYTHONLEGACYWINDOWSSTDIO=utf-8
# 추가 인코딩 안전성 강화
ENV PYTHONUTF8=1
ENV PYTHONHTTPSVERIFY=0
ENV COLUMNS=80
ENV TERM=xterm-256color
ENV PYPPETEER_CHROMIUM_REVISION=""
ENV PYPPETEER_EXECUTABLE_PATH="/usr/bin/chromium"
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates graphviz libfreetype6 libpng16-16 \
    chromium chromium-driver \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 \
    libdrm2 libgtk-3-0 libnspr4 libnss3 libx11-xcb1 libxcomposite1 \
    libxdamage1 libxrandr2 libxss1 libxtst6 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

# Copy project metadata for dependency resolve
COPY pyproject.toml uv.lock ./

# Create and populate venv with uv sync (no dev dependencies)
RUN /root/.local/bin/uv sync --no-dev --frozen

# Copy the rest of the source
COPY . .

# Fix Streamlit static files issue by reinstalling streamlit
RUN /root/.local/bin/uv pip install --force-reinstall streamlit

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8502
ENV DATABASE_PATH=/app/power_demand.db

# Use uv to run streamlit inside managed venv
CMD ["/root/.local/bin/uv", "run", "streamlit", "run", "app.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
