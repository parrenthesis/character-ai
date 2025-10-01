# Character AI Platform Monitoring

This directory contains monitoring configuration for the Character AI Platform production deployment.

## 🚀 Quick Start

```bash
# Start the full monitoring stack
docker-compose up -d

# View services
docker-compose ps

# Check logs
docker-compose logs -f character-ai
```

## 📊 Monitoring Stack

### Services
- **Character AI App**: Main application (port 8000)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards and visualization (port 3000)

### Access URLs
- **Application**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔧 Configuration

### Prometheus (`prometheus.yml`)
- Scrapes metrics from Character AI app every 30s
- Health checks via `/health` endpoint
- Stores metrics in persistent volume

### Grafana
- **Datasource**: Auto-configured Prometheus connection
- **Dashboards**: Character AI Platform dashboard
- **Authentication**: Default admin/admin (change in production)

## 📈 Key Metrics

### Application Health
- `up{job="character-ai"}` - Service availability
- Response time percentiles
- Error rates (4xx, 5xx)

### Performance
- Request rate and latency
- Memory and CPU usage
- Model inference metrics

## 🛠️ Production Setup

### Environment Variables
```bash
# Grafana
GF_SECURITY_ADMIN_PASSWORD=your_secure_password
GF_USERS_ALLOW_SIGN_UP=false

# Prometheus
PROMETHEUS_RETENTION_TIME=30d
```

### Security
- Change default Grafana password
- Use HTTPS in production
- Configure authentication
- Set up alerting rules

## 📝 Custom Dashboards

The Character AI dashboard includes:
- Application health status
- Request rate and response times
- Error rate monitoring
- Performance metrics

## 🔍 Troubleshooting

### Check Service Health
```bash
# Application health
curl http://localhost:8000/health

# Prometheus targets
curl http://localhost:9090/api/v1/targets

# Grafana health
curl http://localhost:3000/api/health
```

### View Logs
```bash
# Application logs
docker-compose logs character-ai

# All services
docker-compose logs
```

### Reset Data
```bash
# Stop and remove volumes
docker-compose down -v

# Restart
docker-compose up -d
```

## 📚 Documentation

- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Character AI API Documentation](../docs/api/)
