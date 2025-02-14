# Base image olarak CUDA destekli PyTorch kullan
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Backend bağımlılıklarını kopyala ve yükle
COPY backend/requirements.txt backend/
RUN pip install -r backend/requirements.txt

# Frontend bağımlılıklarını kopyala ve yükle
COPY frontend/package*.json frontend/
RUN cd frontend && npm install

# Tüm projeyi kopyala
COPY . .

# Frontend'i build et
RUN cd frontend && npm run build

# Backend için port aç
EXPOSE 8000

# Frontend için port aç
EXPOSE 3001

# Başlangıç komutunu ayarla
CMD ["sh", "-c", "cd frontend && npm start & cd backend && python main.py"] 