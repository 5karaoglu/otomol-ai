# Base image olarak CUDA destekli PyTorch kullan
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Timezone ayarı
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Build araçları ve sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Node.js 20.x yükle
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# SSL dizini oluştur
RUN mkdir -p /etc/nginx/ssl

# Çalışma dizinini ayarla
WORKDIR /app

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