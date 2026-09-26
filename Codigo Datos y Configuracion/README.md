# Proyecto Integrador Final — Pipeline híbrido SQL + NoSQL para la detección temprana del riesgo de reprobación

**ESPOCH · Maestría en Estadística, mención Ciencia de Datos e IA · Módulo M1721**
Grupo 1: Alexis Rivera, Lissette Adriano, Shirley Adriano, Cristian Solís · Septiembre 2026

Repositorio: https://github.com/Crisolis517/PIF

## Objetivo

Integrar el registro académico real *Student Performance* (Cortez, 2008; UCI) en PostgreSQL con evidencia
conductual **simulada** (interacciones LMS, tutorías y alertas con texto libre) en MongoDB, construir un
dataset analítico trazable (una fila por matrícula) y estimar el riesgo de reprobación **al cierre del
primer período** sin fuga de información.

> Todo lo que proviene de MongoDB y toda variable con prefijo `sim_` es **simulado** (demostración del
> pipeline). Los hallazgos sustantivos se basan solo en los datos reales de UCI.

## Ejecución con un solo comando

```bash
# 1) Servicios (PostgreSQL 16 + MongoDB 8)
cp .env.example .env            # ajustar credenciales/puertos
docker compose up -d --wait

# 2) Entorno Python (3.9+)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3) Pipeline completo: extract → transform → generate → load → integrate → analysis → benchmark → notebook → tablas
.venv/bin/python run_pipeline.py            # ≈ 90 s en un portátil arm64 de 8 núcleos
.venv/bin/python -m pytest -q               # 17 pruebas (conteos, dominios, fuga, integridad, validadores)

# 4) Informe PDF (requiere tectonic o una distribución LaTeX con xelatex)
.venv/bin/python run_pipeline.py --informe  # o: cd informe && tectonic main.tex
```

`make all` hace lo mismo (pipeline + pruebas + informe). El pipeline es determinista (`SEED = 42`): dos
ejecuciones producen artefactos con idéntico SHA-256 (ver `logs/ultima_ejecucion.json`).

**Sin Docker** (entorno usado para generar este entregable): `scripts/servicios_locales.sh start` crea un clúster PostgreSQL local con `initdb`
(puerto 5433) y `mongod` portable (puerto 27018); basta con apuntar `.env` a esos puertos.

## Estructura

```
proyecto-integrador/
├── run_pipeline.py            # punto de entrada único
├── docker-compose.yml         # PostgreSQL 16 + MongoDB 8
├── requirements.txt           # versiones fijadas (requirements-lock.txt = pip freeze completo)
├── .env.example
├── data/raw/                  # student-mat.csv, student-por.csv (UCI) + synthetic/ (JSON crudos con defectos inyectados)
├── data/staging/              # intermedios del ETL (tablas 3FN, identidad, documentos, cuarentena)
├── data/processed/            # dataset_analitico.parquet/.csv, diccionario_datos.csv, linaje.csv
├── sql/                       # 01 DDL 3FN · 02 vista + rol · 03 integridad · 04 analíticas
├── mongo/                     # validadores $jsonSchema+$expr, índices, pipelines de agregación
├── src/                       # config, tracking, extract, transform, generate_synthetic,
│                              # transform_synthetic, load, integrate, analysis, benchmark_escalabilidad, report_tables
├── tests/                     # pytest
├── notebooks/analisis.ipynb   # evidencias en vivo (SQL, Mongo) y reproducción del modelo
├── results/                   # salidas de consultas, pipelines, pruebas estadísticas y modelos
├── logs/                      # conteos por etapa, reglas de calidad, checksums
└── informe/                   # main.tex, generated/ (tablas y macros generadas), figuras/, PDF final
```

## Cifras clave (salidas del pipeline)

| Concepto | Valor |
|---|---|
| Filas UCI | 395 (MAT) + 649 (POR) = 1 044 matrículas |
| Estudiantes únicos (regla de identidad en 2 etapas) | 674 (370 en ambas asignaturas) |
| Calificaciones (1FN) | 3 132 |
| Documentos MongoDB | 2 022 `seguimiento_riesgo` + 282 `alertas_riesgo` (100 % de estudiantes) |
| Dataset analítico | 1 044 filas × 58 columnas; `en_riesgo` = 230 (22,0 %) |
| Modelo temprano (corte P1, regresión logística) | AUC-ROC test 0,948; recall 0,902 |

## Documentación

El informe (`informe/Proyecto_Integrador_Final_Grupo1.pdf`) contiene la arquitectura, el catálogo de
reglas R-01…R-26, el linaje, el diccionario de datos, los resultados y las limitaciones. Todas las
cifras del informe se generan desde `results/` y `logs/` mediante `src/report_tables.py`.

## Licencia y datos

Dataset UCI bajo CC BY 4.0 (Cortez, 2008, https://doi.org/10.24432/C5TG7T). Los datos simulados no
contienen información de personas reales; tutores y orientadores se identifican con códigos (`TUT-xx`, `ORI-xx`).
