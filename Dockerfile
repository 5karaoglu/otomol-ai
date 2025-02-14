# Base image olarak CUDA destekli PyTorch kullan
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Timezone ayarı
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Çalışma dizinini ayarla
WORKDIR /app

# Node.js 20.x ve diğer sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && npm install -g serve \
    && rm -rf /var/lib/apt/lists/*

# Backend bağımlılıklarını kopyala ve yükle
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Frontend bağımlılıklarını kopyala ve yükle
COPY frontend/package*.json frontend/
RUN cd frontend && npm install

# Tüm kaynak kodlarını kopyala
COPY . .

# Frontend'i build et
RUN cd frontend && npm run build

# Başlangıç scriptini oluştur
RUN echo '#!/bin/bash\ncd /app/backend && python main.py &\ncd /app/frontend && serve -s build -l 3001' > /app/start.sh && \
    chmod +x /app/start.sh

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Environment variable for React
ENV PORT=3001
ENV HOST=0.0.0.0

# Başlangıç komutu
CMD ["/app/start.sh"] 