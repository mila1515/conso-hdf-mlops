.PHONY: install lock ingest train predict backfill monitor daily weekly schedule notify-test api dashboard mlflow test lint docker-up docker-down

install:
	pip install -r requirements-dev.txt && pip install --no-deps -e .
	@test -f .env || cp .env.example .env

# Régénère les versions figées (après modification de requirements.txt)
lock:
	uv pip compile requirements.txt --python-version 3.11 --python-platform linux -o requirements.lock

ingest:
	python -m conso_hdf.pipeline ingest

train:
	python -m conso_hdf.pipeline train

predict:
	python -m conso_hdf.pipeline predict

monitor:
	python -m conso_hdf.pipeline monitor

backfill:
	python -m conso_hdf.pipeline backfill

daily:
	python -m conso_hdf.pipeline daily

weekly:
	python -m conso_hdf.pipeline weekly

schedule:
	python -m conso_hdf.pipeline schedule

notify-test:
	python -m conso_hdf.pipeline notify-test

api:
	uvicorn conso_hdf.api.main:app --reload --port 8700

dashboard:
	streamlit run dashboard/app.py --server.port 8701

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8702

test:
	pytest -q

lint:
	ruff check src tests dashboard

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
