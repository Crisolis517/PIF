"""Configuración central del pipeline: rutas, semilla, conexiones y calendario lectivo.

Toda constante que afecte a los resultados vive aquí, para que una única
modificación quede trazada y sea reproducible.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
SEED = 42  # semilla global: generación sintética, partición y modelos

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
DATA_RAW = ROOT / "data" / "raw"
DATA_SYNTH_RAW = DATA_RAW / "synthetic"          # JSON "crudo" con defectos inyectados
DATA_STAGING = ROOT / "data" / "staging"         # intermedios del ETL
DATA_PROCESSED = ROOT / "data" / "processed"     # dataset analítico final
RESULTS = ROOT / "results"                       # tablas y métricas del análisis
FIGURES = ROOT / "informe" / "figuras"
GENERATED_TEX = ROOT / "informe" / "generated"   # tablas/macros LaTeX generados por código
LOGS = ROOT / "logs"
SQL_DIR = ROOT / "sql"
MONGO_DIR = ROOT / "mongo"

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
UCI_FILES = {"MAT": DATA_RAW / "student-mat.csv", "POR": DATA_RAW / "student-por.csv"}
# SHA-256 de los CSV oficiales (verificados en la descarga del 26/09/2026)
UCI_SHA256 = {
    "MAT": "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80",
    "POR": "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
}

# --------------------------------------------------------------------------- #
# Conexiones (valores por defecto = docker-compose.yml)
# --------------------------------------------------------------------------- #
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "pi_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DB = os.getenv("PG_DB", "proyecto_integrador")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "proyecto_integrador")


def pg_url() -> str:
    """URL SQLAlchemy de PostgreSQL (driver psycopg 3)."""
    auth = PG_USER if not PG_PASSWORD else f"{PG_USER}:{PG_PASSWORD}"
    return f"postgresql+psycopg://{auth}@{PG_HOST}:{PG_PORT}/{PG_DB}"


# --------------------------------------------------------------------------- #
# Dominio del problema
# --------------------------------------------------------------------------- #
UMBRAL_APROBACION = 10  # escala portuguesa 0–20: aprobado si nota >= 10

# Calendario lectivo 2005–2006 (aproximación declarada del calendario portugués).
# Mapeo explícito: P1 <-> G1, P2 <-> G2, P3 <-> G3 (nota final).
PERIODOS = {
    "P1": {"orden": 1, "nota": "G1", "inicio": date(2005, 9, 15), "fin": date(2005, 12, 16),
           "nombre": "1.er período (sep–dic 2005)"},
    "P2": {"orden": 2, "nota": "G2", "inicio": date(2006, 1, 3), "fin": date(2006, 3, 24),
           "nombre": "2.º período (ene–mar 2006)"},
    "P3": {"orden": 3, "nota": "G3", "inicio": date(2006, 4, 10), "fin": date(2006, 6, 16),
           "nombre": "3.er período / nota final (abr–jun 2006)"},
}
CORTE_TEMPRANO = "P1"  # la predicción temprana se emite al cierre de P1

ESCUELAS = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}
ASIGNATURAS = {"MAT": "Matemáticas", "POR": "Lengua Portuguesa"}

# Regla de identidad (Cortez, 2008): 13 atributos de la clave oficial UCI
CLAVE_IDENTIDAD = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu",
                   "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
# Atributos personales (no dependen de la asignatura) usados para desambiguar
ATRIBUTOS_DESAMBIGUACION = ["guardian", "traveltime", "studytime", "schoolsup", "famsup",
                            "activities", "higher", "romantic", "famrel", "freetime",
                            "goout", "Dalc", "Walc", "health"]
UMBRAL_DESAMBIGUACION = 12  # coincidencias mínimas (de 14) para aceptar un emparejamiento
# Atributos que dependen de la asignatura -> van a la tabla matricula
ATRIBUTOS_MATRICULA = ["failures", "paid", "absences"]


def ensure_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    for d in (DATA_SYNTH_RAW, DATA_STAGING, DATA_PROCESSED, RESULTS, FIGURES,
              GENERATED_TEX, LOGS):
        d.mkdir(parents=True, exist_ok=True)
