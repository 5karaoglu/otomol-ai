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
    procps \
    net-tools \
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
ENV REACT_APP_BACKEND_URL=http://213.181.123.11:8000

# Frontend'i build et
RUN cd frontend && REACT_APP_BACKEND_URL=http://213.181.123.11:8000 npm run build

# Log dosyası oluştur
RUN mkdir -p /var/log/otomol && touch /var/log/otomol/app.log

# Başlangıç scriptini oluştur
RUN echo '#!/bin/bash\n\
\n\
# Log dosyası\n\
LOG_FILE="/var/log/otomol/app.log"\n\
\n\
# Timestamp fonksiyonu\n\
timestamp() {\n\
    date "+%Y-%m-%d %H:%M:%S"\n\
}\n\
\n\
# Loglama fonksiyonu\n\
log() {\n\
    echo "$(timestamp) - $1" | tee -a $LOG_FILE\n\
}\n\
\n\
# Hata loglama fonksiyonu\n\
log_error() {\n\
    echo "$(timestamp) - HATA: $1" | tee -a $LOG_FILE >&2\n\
}\n\
\n\
# Sistem durumunu kontrol et\n\
check_system() {\n\
    log "Sistem durumu kontrol ediliyor..."\n\
    \n\
    # RAM kontrolü\n\
    local total_mem=$(free -m | awk \'/Mem:/ {print $2}\')\n\
    local used_mem=$(free -m | awk \'/Mem:/ {print $3}\')\n\
    local mem_usage=$((used_mem * 100 / total_mem))\n\
    log "Bellek kullanımı: ${mem_usage}%"\n\
    \n\
    # Disk kontrolü\n\
    local disk_usage=$(df -h / | awk \'/\\// {print $5}\' | tr -d "%")\n\
    log "Disk kullanımı: ${disk_usage}%"\n\
    \n\
    # Port kontrolü\n\
    if netstat -tuln | grep -q ":8000 "; then\n\
        log_error "8000 portu zaten kullanımda!"\n\
        return 1\n\
    fi\n\
    if netstat -tuln | grep -q ":3001 "; then\n\
        log_error "3001 portu zaten kullanımda!"\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Backend başlatma fonksiyonu\n\
start_backend() {\n\
    cd /app/backend\n\
    log "Backend başlatılıyor..."\n\
    \n\
    # Python sürümünü kontrol et\n\
    python --version >> $LOG_FILE 2>&1\n\
    \n\
    # Backend\'i başlat\n\
    python main.py > >(tee -a $LOG_FILE) 2> >(tee -a $LOG_FILE >&2) &\n\
    BACKEND_PID=$!\n\
    \n\
    # Backend\'in başlamasını bekle\n\
    log "Backend\'in hazır olması bekleniyor..."\n\
    local counter=0\n\
    while ! netstat -tuln | grep -q ":8000 "; do\n\
        sleep 1\n\
        ((counter++))\n\
        if [ $counter -ge 30 ]; then\n\
            log_error "Backend 30 saniye içinde başlatılamadı!"\n\
            return 1\n\
        fi\n\
        if ! ps -p $BACKEND_PID > /dev/null; then\n\
            log_error "Backend process\'i beklenmedik şekilde sonlandı!"\n\
            return 1\n\
        fi\n\
    done\n\
    \n\
    log "Backend başarıyla başlatıldı (PID: $BACKEND_PID)"\n\
    return 0\n\
}\n\
\n\
# Frontend başlatma fonksiyonu\n\
start_frontend() {\n\
    cd /app/frontend\n\
    log "Frontend başlatılıyor..."\n\
    \n\
    # Node.js sürümünü kontrol et\n\
    node --version >> $LOG_FILE 2>&1\n\
    \n\
    # Frontend\'i başlat\n\
    serve -s build -l 3001 --host 0.0.0.0 > >(tee -a $LOG_FILE) 2> >(tee -a $LOG_FILE >&2)\n\
}\n\
\n\
# Ana çalıştırma fonksiyonu\n\
main() {\n\
    log "Uygulama başlatılıyor..."\n\
    \n\
    # Sistem kontrolü\n\
    if ! check_system; then\n\
        log_error "Sistem kontrolleri başarısız!"\n\
        exit 1\n\
    fi\n\
    \n\
    # Backend\'i başlat\n\
    if ! start_backend; then\n\
        log_error "Backend başlatılamadı!"\n\
        exit 1\n\
    fi\n\
    \n\
    # Frontend\'i başlat\n\
    start_frontend\n\
}\n\
\n\
# Uygulamayı başlat\n\
main\n\
' > /app/start.sh && chmod +x /app/start.sh

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Başlangıç komutu
CMD ["/app/start.sh"] 