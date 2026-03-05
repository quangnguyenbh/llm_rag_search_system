.PHONY: dev test lint format docker-up docker-down ingest crawl

dev:
	uvicorn src.main:app --reload --port 8000

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

docker-up:
	docker compose up -d

docker-down:
	docker compose down

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(msg)"

ingest:
	python -m scripts.bulk_ingest $(args)

crawl-ia:
	python -m scripts.crawl_internet_archive $(args)

worker:
	@echo "Celery worker is not configured for this project. Please update the 'worker' target in the Makefile once a Celery app is available."
