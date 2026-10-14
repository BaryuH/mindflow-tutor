.PHONY: install db backend client lint test build

TORCH_CPU := --extra-index-url https://download.pytorch.org/whl/cpu

install: ## Install backend (CPU torch) and frontend dependencies
	cd backend && pip install $(TORCH_CPU) -r requirements-web.txt -r requirements-ai.txt ruff pytest pytest-asyncio
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
