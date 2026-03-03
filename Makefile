.PHONY: setup train preprocess test lint format clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo "Run 'source .venv/bin/activate' to activate the environment"

train:
	python scripts/train.py $(ARGS)

preprocess:
	python scripts/preprocess.py $(ARGS)

eval:
	python scripts/eval.py $(ARGS)

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	rm -rf outputs/ multirun/ wandb/ .hydra/
	find . -type d -name __pycache__ -exec rm -rf {} +
