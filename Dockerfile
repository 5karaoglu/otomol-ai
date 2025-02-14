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

# Environment variable for React and Backend
ENV PORT=3001
ENV HOST=0.0.0.0
ENV REACT_APP_BACKEND_URL=http://90.84.225.124:8000

# Frontend'i build et
RUN cd frontend && REACT_APP_BACKEND_URL=http://90.84.225.124:8000 npm run build

# Başlangıç scriptini oluştur
RUN echo '#!/bin/bash\n\
cd /app/backend\n\
echo "Starting backend..."\n\
python main.py > /proc/1/fd/1 2>/proc/1/fd/2 &\n\
BACKEND_PID=$!\n\
echo "Backend started with PID: $BACKEND_PID"\n\
sleep 5\n\
if ps -p $BACKEND_PID > /dev/null; then\n\
    echo "Backend is running successfully"\n\
    cd /app/frontend\n\
    echo "Starting frontend..."\n\
    exec serve -s build -l 3001\n\
else\n\
    echo "Backend failed to start"\n\
    exit 1\n\
fi' > /app/start.sh && \
    chmod +x /app/start.sh

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Başlangıç komutu
CMD ["/app/start.sh"] 