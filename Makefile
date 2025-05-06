.PHONY: setup build activate install test style bumpver bumpver-minor

print:
	echo $(SHELL)

all: setup install

setup:
	pip install uv
	uv venv

build:
	uv build

install:
	uv sync

xpu:
	uv sync --extra xpu
	uv pip install torch==2.7.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu

style:
	uv run pre-commit run --all-files

bumpver:
	uv run python -m bumpver update -n --patch

bumpver-minor:
	uv run python -m bumpver update -n
