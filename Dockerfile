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
    supervisor \
    nginx \
    certbot \
    python3-certbot-nginx \
    openssl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

# SSL sertifikası oluştur
RUN mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/nginx.key \
    -out /etc/nginx/ssl/nginx.crt \
    -subj "/C=TR/ST=Istanbul/L=Istanbul/O=Otomol/CN=213.181.123.11"

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
ENV REACT_APP_BACKEND_URL=http://213.181.123.11:54722

# Frontend'i build et
RUN cd frontend && \
    REACT_APP_BACKEND_URL=http://213.181.123.11:54722 npm run build && \
    rm -rf /var/www/html/* && \
    cp -r build/* /var/www/html/

# Supervisor yapılandırması
RUN mkdir -p /var/log/supervisor && \
    mkdir -p /etc/supervisor/conf.d && \
    mkdir -p /var/run/supervisor && \
    ln -s /var/run /run && \
    echo '[supervisord]\n\
nodaemon=true\n\
logfile=/var/log/supervisor/supervisord.log\n\
pidfile=/var/run/supervisor/supervisord.pid\n\
childlogdir=/var/log/supervisor\n\
user=root\n\
\n\
[unix_http_server]\n\
file=/var/run/supervisor.sock\n\
chmod=0700\n\
\n\
[rpcinterface:supervisor]\n\
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface\n\
\n\
[supervisorctl]\n\
serverurl=unix:///var/run/supervisor.sock\n\
\n\
[program:backend]\n\
command=python /app/backend/main.py\n\
directory=/app/backend\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/supervisor/backend.err.log\n\
stdout_logfile=/var/log/supervisor/backend.out.log\n\
\n\
[program:frontend]\n\
command=nginx -g "daemon off;"\n\
directory=/app/frontend\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/supervisor/frontend.err.log\n\
stdout_logfile=/var/log/supervisor/frontend.out.log\n\
' > /etc/supervisor/conf.d/supervisord.conf

# Nginx yapılandırması
RUN echo 'server {\n\
    listen 3001 ssl;\n\
    server_name _;\n\
    \n\
    ssl_certificate /etc/nginx/ssl/nginx.crt;\n\
    ssl_certificate_key /etc/nginx/ssl/nginx.key;\n\
    \n\
    root /var/www/html;\n\
    index index.html;\n\
    \n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Supervisor ile servisleri başlat
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 