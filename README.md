### /Users/atomasini/Development/MLX-model-manager/./README.md
```markdown
1: # MLX Manager
2: 
3: A web-based application for managing MLX-optimized language models on Apple Silicon Macs.
4: 
5: ## Features
6: 
7: - **Model Browser**: Search and download MLX models from HuggingFace's mlx-community
8: - **Server Profile Manager**: Create and manage mlx-openai-server configurations
9: - **Instance Controller**: Start/stop servers with real-time health monitoring
10: - **launchd Integration**: Configure models as macOS system services
11: 
12: ## Requirements
13: 
14: - macOS with Apple Silicon (M1/M2/M3/M4)
15: - Python 3.11+
16: - Node.js 20+
17: - mlx-openai-server (`pip install mlx-openai-server`)
18: 
19: ## Quick Start
20: 
21: ### Development
22: 
23: ```bash
24: # Start both backend and frontend
25: ./scripts/dev.sh
26: 
27: # Or start individually:
28: 
29: # Backend (terminal 1)
30: cd backend
31: python -m venv .venv
32: source .venv/bin/activate
33: pip install -e ".[dev]"
34: uvicorn app.main:app --reload --port 8080
35: 
36: # Frontend (terminal 2)
37: cd frontend
38: npm install
39: npm run dev
40: ```
41: 
42: ### Access
43: 
44: - Frontend: http://localhost:5173
45: - Backend API: http://localhost:8080
46: - API Documentation: http://localhost:8080/docs
47: 
48: ## Project Structure
49: 
50: ```
51: mlx-manager/
52: ├── backend/           # FastAPI backend
53: │   ├── app/
54: │   │   ├── routers/   # API endpoints
55: │   │   ├── services/  # Business logic
56: │   │   └── models.py  # Database models
57: │   └── pyproject.toml
58: ├── frontend/          # SvelteKit frontend
59: │   ├── src/
60: │   │   ├── lib/       # Components, stores, API
61: │   │   └── routes/    # Pages
62: │   └── package.json
63: └── scripts/           # Development scripts
64: ```
65: 
66: ## License
67: 
68: MIT
```
