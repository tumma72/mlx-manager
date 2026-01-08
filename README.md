# MLX Manager

A web-based application for managing MLX-optimized language models on Apple Silicon Macs.

## Features

- **Model Browser**: Search and download MLX models from HuggingFace's mlx-community
- **Server Profile Manager**: Create and manage mlx-openai-server configurations
- **Instance Controller**: Start/stop servers with real-time health monitoring
- **launchd Integration**: Configure models as macOS system services

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 20+
- mlx-openai-server (`pip install mlx-openai-server`)

## Quick Start

### Development

```bash
# Start both backend and frontend
./scripts/dev.sh

# Or start individually:

# Backend (terminal 1)
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8080

# Frontend (terminal 2)
cd frontend
npm install
npm run dev
```

### Access

- Frontend: http://localhost:5173
- Backend API: http://localhost:8080
- API Documentation: http://localhost:8080/docs

## Project Structure

```
mlx-manager/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── routers/   # API endpoints
│   │   ├── services/  # Business logic
│   │   └── models.py  # Database models
│   └── pyproject.toml
├── frontend/          # SvelteKit frontend
│   ├── src/
│   │   ├── lib/       # Components, stores, API
│   │   └── routes/    # Pages
│   └── package.json
└── scripts/           # Development scripts
```

## License

MIT
