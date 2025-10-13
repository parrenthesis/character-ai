# Production Deployment Guide

## Overview

This guide covers deploying the Character AI in production environments with enterprise-grade security, monitoring, and scalability.

## Production Architecture

### System Requirements

#### Minimum Production Requirements
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **CPU**: 8 cores (Intel Xeon or AMD EPYC)
- **Network**: 1Gbps connection
- **OS**: Ubuntu 20.04 LTS or CentOS 8+

#### Recommended Production Requirements
- **Memory**: 32GB RAM
- **Storage**: 500GB NVMe SSD
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **Network**: 10Gbps connection
- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA A100 or V100 (optional, for LLM acceleration)

### Security Configuration

#### Environment Variables
```bash
# Required security variables
export CAI_JWT_SECRET="your-super-secure-jwt-secret-here"
export CAI_PRIVATE_KEY_FILE="/etc/cai/keys/private.pem"
export CAI_PUBLIC_KEY_FILE="/etc/cai/keys/public.pem"
export CAI_REQUIRE_HTTPS="true"
export CAI_RATE_LIMIT_REQUESTS_PER_MINUTE="1000"
export CAI_RATE_LIMIT_BURST="100"

# Optional security variables
export CAI_ENABLE_DEVICE_REGISTRATION="true"
export CAI_JWT_EXPIRY_SECONDS="3600"
export CAI_JWT_ALGORITHM="HS256"

# Required for PyTorch 2.8+ compatibility with XTTS v2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

> **Note**: The `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` environment variable is required for PyTorch 2.8+ compatibility with Coqui XTTS v2 models. This prevents CUDA context conflicts during model loading.

#### SSL/TLS Configuration
```bash
# Generate SSL certificates
sudo openssl req -x509 -newkey rsa:4096 -keyout /etc/cai/ssl/private.key \
  -out /etc/cai/ssl/certificate.crt -days 365 -nodes

# Configure nginx with SSL
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/cai/ssl/certificate.crt;
    ssl_certificate_key /etc/cai/ssl/private.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### 1. Create Dockerfile
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 cai && chown -R cai:cai /app
USER cai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.character.ai.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  cai-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CAI_JWT_SECRET=${CAI_JWT_SECRET}
      - CAI_REQUIRE_HTTPS=true
    volumes:
      - ./configs:/app/configs
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - cai-app
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

#### 3. Deploy with Docker Compose
```bash
# Build and start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cai-app
```

### Method 2: Kubernetes Deployment

#### 1. Create Kubernetes manifests

**namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cai-production
```

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cai-app
  namespace: cai-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cai-app
  template:
    metadata:
      labels:
        app: cai-app
    spec:
      containers:
      - name: cai-app
        image: your-registry/cai:latest
        ports:
        - containerPort: 8000
        env:
        - name: CAI_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: cai-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: cai-service
  namespace: cai-production
spec:
  selector:
    app: cai-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**ingress.yaml**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cai-ingress
  namespace: cai-production
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: cai-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cai-service
            port:
              number: 80
```

#### 2. Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f namespace.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n cai-production
kubectl get services -n cai-production
kubectl get ingress -n cai-production
```

### Method 3: Traditional Server Deployment

#### 1. System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install system dependencies
sudo apt install build-essential libsndfile1 ffmpeg nginx redis-server

# Create application user
sudo useradd -m -s /bin/bash cai
sudo mkdir -p /opt/cai
sudo chown cai:cai /opt/cai
```

#### 2. Application Deployment
```bash
# Switch to cai user
sudo su - cai

# Clone repository
cd /opt/cai
git clone https://github.com/your-org/character-ai.git
cd character-ai

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --no-dev

# Create configuration
mkdir -p configs
cp .env.example .env
# Edit .env with production values
```

#### 3. Systemd Service
```bash
# Create systemd service file
sudo tee /etc/systemd/system/cai.service > /dev/null <<EOF
[Unit]
Description=Character AI
After=network.target

[Service]
Type=simple
User=cai
WorkingDirectory=/opt/cai/character-ai
ExecStart=/opt/cai/character-ai/.venv/bin/python -m uvicorn src.character.ai.web.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cai
sudo systemctl start cai
sudo systemctl status cai
```

## Monitoring and Observability

### 1. Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Metrics endpoint
curl http://localhost:8000/metrics
```

### 2. Logging Configuration
```python
# Configure structured logging
import logging
from src.character.ai.core.logging import get_logger

logger = get_logger(__name__)

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/cai/app.log'),
        logging.StreamHandler()
    ]
)
```

### 3. Monitoring Stack
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  kibana:
    image: kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  grafana_data:
```

## Performance Optimization

### 1. Production Configuration
```python
# production_config.py
from src.character.ai.core.config import Config, Environment

# Production-optimized configuration
production_config = Config(
    environment=Environment.PRODUCTION,
    # Performance settings
    max_cpu_threads=8,
    enable_cpu_limiting=True,
    # Memory settings
    max_memory_usage_gb=16.0,
    # Security settings
    require_https=True,
    rate_limit_requests_per_minute=1000,
    # Monitoring settings
    enable_performance_monitoring=True,
    monitoring_interval_seconds=30
)
```

### 2. Load Balancing
```nginx
# nginx.conf
upstream cai_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://cai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Recovery

### 1. Database Backup
```bash
# Backup character data
cp -r /opt/cai/characters /backup/characters-$(date +%Y%m%d)

# Backup configuration
cp -r /opt/cai/configs /backup/configs-$(date +%Y%m%d)

# Backup logs
cp -r /var/log/cai /backup/logs-$(date +%Y%m%d)
```

### 2. Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/cai-$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup application data
cp -r /opt/cai/characters $BACKUP_DIR/
cp -r /opt/cai/configs $BACKUP_DIR/
cp -r /var/log/cai $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# Upload to cloud storage (optional)
# aws s3 cp $BACKUP_DIR.tar.gz s3://your-backup-bucket/
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Restart service if needed
sudo systemctl restart cai
```

#### 2. High CPU Usage
```bash
# Check CPU usage
top
htop

# Check for stuck processes
ps aux | grep python
```

#### 3. Network Issues
```bash
# Check network connectivity
curl -I http://localhost:8000/health
netstat -tlnp | grep :8000

# Check firewall
sudo ufw status
```

### Log Analysis
```bash
# View application logs
tail -f /var/log/cai/app.log

# Search for errors
grep -i error /var/log/cai/app.log

# Monitor real-time logs
journalctl -u cai -f
```

## Security Checklist

- [ ] JWT secret is properly configured
- [ ] SSL/TLS certificates are valid
- [ ] Rate limiting is enabled
- [ ] Device registration is configured
- [ ] Private keys are secured
- [ ] HTTPS is enforced
- [ ] Firewall rules are configured
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Access logging enabled

## Production Readiness Checklist

- [ ] All tests passing (98/98)
- [ ] Security hardening complete
- [ ] Performance optimization applied
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures
- [ ] Documentation complete
- [ ] SSL/TLS certificates installed
- [ ] Load balancing configured
- [ ] Health checks implemented
- [ ] Log aggregation setup
