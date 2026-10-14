# FaceGuard — reproducible pipeline. `make all` = assets -> data -> train -> eval -> report
PY ?= python
export DATABASE_URL ?= sqlite:///./faceguard.db
export LOCAL_ARTIFACT_DIR ?= ./model_artifacts/

.PHONY: all setup assets data sessions train register eval report test lint run up down clean

all: assets data sessions train eval report

setup:            ## install serving + training dependencies
	$(PY) -m pip install -r requirements-train.txt

assets:           ## face detector/recogniser weights + public face photos
	./scripts/download_assets.sh

data:             ## bona fide corpus: detect, crop, dedupe, split
	$(PY) -m src.training.data --src data/raw/face-mask-detection/dataset/without_mask --src data/raw/deepface/tests/unit/dataset

sessions:         ## adversarial campaigns (train / val / test)
	$(PY) -m src.evaluation.harness --split train
	$(PY) -m src.evaluation.harness --split val
	$(PY) -m src.evaluation.harness --split test

train:            ## PAD + capture-integrity (sklearn) and deepfake CNN (PyTorch -> ONNX)
	$(PY) -m src.training.train_tabular
	$(PY) -m src.training.train_deepfake --epochs 30

eval:             ## score, calibrate policy on val, evaluate on test
	$(PY) -m src.evaluation.run_eval

report:           ## charts + docs/02 test report from reports/results.json
	$(PY) -m src.evaluation.report

register:         ## publish detectors to the MSP registry and promote (in-process)
	$(PY) -m src.training.register --version 1.0.0 --in-process

test:
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .

run: register     ## local API on :8000 (SQLite), demo at /demo
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

up:               ## full stack: API + PostgreSQL + Prometheus + Grafana
	docker compose up --build -d

down:
	docker compose down -v

clean:
	rm -rf data/sessions reports/scores_*.csv model_artifacts faceguard.db
