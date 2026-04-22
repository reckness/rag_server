# ============================================================
# rag-service Docker 镜像（单阶段构建）
# ============================================================
FROM python:3.10-slim-bookworm

WORKDIR /app

# ---- 系统依赖（含 LibreOffice 用于文档转 PDF）----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libfreetype6 \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 项目代码 ----
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
