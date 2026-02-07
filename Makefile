.PHONY: dev test docker-up docker-down clean install lint

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

docker-build:
	docker build -t onemin-openai-proxy .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	python -m py_compile app/*.py tests/*.py
