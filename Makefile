.PHONY: install db backend client lint test build

# PyTorch wheel index: cpu, or cu124 on the GPU host (make install TORCH_INDEX=cu124)
TORCH_INDEX ?= cpu
TORCH := --extra-index-url https://download.pytorch.org/whl/$(TORCH_INDEX)

install: ## Install backend and frontend dependencies (TORCH_INDEX=cpu|cu124)
	cd backend && pip install $(TORCH) -r requirements-web.txt -r requirements-ai.txt ruff pytest pytest-asyncio
	cd client && npm ci

db: ## Start MongoDB in Docker
	docker compose up -d mongo

backend: ## Run the API on :8000 with reload
	cd backend && uvicorn server:app --reload --port 8000

client: ## Run the Vite dev server on :5173
	cd client && npm run dev

lint: ## Lint + typecheck both apps
	cd backend && ruff check .
	cd client && npm run lint && npm run typecheck

test: ## Run all tests
	cd backend && pytest
	cd client && npm test

build: ## Build the production image
	docker build -f Dockerfile.prod -t mindflow-tutor .
