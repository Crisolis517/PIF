# Uso: make all  (servicios + pipeline + pruebas + informe)
PY ?= .venv/bin/python

.PHONY: all venv servicios pipeline test informe limpiar

all: pipeline test informe

venv:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

servicios:
	docker compose up -d --wait

pipeline:
	$(PY) run_pipeline.py

test:
	$(PY) -m pytest -q | tee logs/pytest_ultima_ejecucion.txt

informe:
	cd informe && tectonic main.tex && cp main.pdf Proyecto_Integrador_Final_Grupo1.pdf

limpiar:
	rm -rf data/staging data/processed results informe/generated logs/*.json logs/*.csv
