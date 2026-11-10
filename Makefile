PY ?= .venv/bin/python
LIB_NIX := $(shell for d in /nix/store/*-gcc-*-lib; do [ -e $$d/lib/libstdc++.so.6 ] && echo $$d/lib && break; done 2>/dev/null)
export LD_LIBRARY_PATH := $(LIB_NIX):/run/opengl-driver/lib:$(LD_LIBRARY_PATH)
export PYTHONHASHSEED := 0
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: setup dados treinar eval erros teste all
setup:
	python3 -m venv .venv && $(PY) -m pip install -q torch transformers peft accelerate pytest

dados:
	$(PY) -m src.dados

treinar:
	for s in 1 2 3; do $(PY) -m src.treinar --saida adaptadores/semente$$s --semente $$s; done

eval:
	$(PY) -m evals.run_eval --adaptadores adaptadores/semente1 adaptadores/semente2 adaptadores/semente3

erros:
	$(PY) -m evals.analise_erros

teste:
	DISPOSITIVO=cpu $(PY) -m pytest -q -p no:cacheprovider

all: dados treinar eval erros teste
