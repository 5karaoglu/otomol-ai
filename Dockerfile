# Base image olarak CUDA destekli PyTorch kullan
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Timezone ayarı
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python 3.10 ve pip kur
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    curl \
    ffmpeg \
    git \
    procps \
    net-tools \
    supervisor \
    nginx \
    certbot \
    python3-certbot-nginx \
    openssl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20.x yükle
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# SSL dizini oluştur
RUN mkdir -p /etc/nginx/ssl

# Çalışma dizinini ayarla
WORKDIR /app

# Pip'i güncelle ve wheel kur
RUN python3 -m pip install --upgrade pip && \
    pip install wheel

# Backend bağımlılıklarını kopyala ve yükle
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend bağımlılıklarını kopyala ve yükle
COPY frontend/package*.json frontend/
RUN cd frontend && npm install

# Tüm kaynak kodlarını kopyala
COPY . .

# Environment variable for React and Backend
ENV PORT=3001
ENV HOST=0.0.0.0
ENV REACT_APP_BACKEND_URL=http://213.181.123.11:54722

# CUDA ve PyTorch environment değişkenleri
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Frontend'i build et
RUN cd frontend && \
    REACT_APP_BACKEND_URL=http://213.181.123.11:54722 npm run build && \
    rm -rf /var/www/html/* && \
    cp -r build/* /var/www/html/

# Model cache dizini oluştur
RUN mkdir -p /app/backend/model_cache && chmod 777 /app/backend/model_cache

# Supervisor yapılandırması
COPY supervisord.conf /etc/supervisor/conf.d/

# Nginx yapılandırması
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Supervisor ile servisleri başlat
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 