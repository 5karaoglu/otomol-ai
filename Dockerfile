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

# SSL dizini oluştur
RUN mkdir -p /etc/nginx/ssl

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
command=python /app/backend/main.py --port 8001\n\
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
    listen 3001;\n\
    server_name _;\n\
    \n\
    root /var/www/html;\n\
    index index.html;\n\
    \n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n\
\n\
server {\n\
    listen 8000;\n\
    server_name _;\n\
    \n\
    location / {\n\
        proxy_pass http://localhost:8001;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        proxy_read_timeout 300;\n\
        proxy_connect_timeout 300;\n\
        proxy_send_timeout 300;\n\
    }\n\
    \n\
    location /ws {\n\
        proxy_pass http://localhost:8001/ws;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        proxy_read_timeout 300;\n\
        proxy_connect_timeout 300;\n\
        proxy_send_timeout 300;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

# Port'ları aç
EXPOSE 8000
EXPOSE 3001

# Supervisor ile servisleri başlat
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 