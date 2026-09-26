"""EXTRACT — adquisición y lectura del registro académico UCI (fuente F1).

- Descarga el ZIP oficial si los CSV no existen y verifica su SHA-256.
- Lee cada CSV (separador ';') con esquema explícito de 33 columnas.
- Añade columnas de linaje (`archivo_origen`, `fila_origen`) y deja una copia
  inmutable en `data/staging/uci_raw.parquet`.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from config import DATA_RAW, DATA_STAGING, UCI_FILES, UCI_SHA256, UCI_URL, ensure_dirs
from tracking import artefacto, etapa, get_logger, regla, sha256

# Esquema esperado (documentación UCI, student.txt): nombre -> tipo lógico
ESQUEMA_UCI = {
    "school": "cat", "sex": "cat", "age": "int", "address": "cat", "famsize": "cat",
    "Pstatus": "cat", "Medu": "int", "Fedu": "int", "Mjob": "cat", "Fjob": "cat",
    "reason": "cat", "guardian": "cat", "traveltime": "int", "studytime": "int",
    "failures": "int", "schoolsup": "cat", "famsup": "cat", "paid": "cat",
    "activities": "cat", "nursery": "cat", "higher": "cat", "internet": "cat",
    "romantic": "cat", "famrel": "int", "freetime": "int", "goout": "int", "Dalc": "int",
    "Walc": "int", "health": "int", "absences": "int", "G1": "int", "G2": "int", "G3": "int",
}


def descargar_uci() -> None:
    """Descarga y descomprime el dataset si falta algún CSV (ZIP anidado en UCI)."""
    if all(p.exists() for p in UCI_FILES.values()):
        return
    import urllib.request
    log = get_logger()
    log.info("Descargando dataset UCI desde %s", UCI_URL)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = [n for n in outer.namelist() if n.endswith(".zip")][0]
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    inner.extractall(DATA_RAW)


def verificar_checksums() -> dict[str, str]:
    """Verifica que los CSV coinciden con la versión oficial registrada."""
    hashes = {k: sha256(p) for k, p in UCI_FILES.items()}
    for k, h in hashes.items():
        if h != UCI_SHA256[k]:
            raise ValueError(f"Checksum distinto para {k}: {h} (esperado {UCI_SHA256[k]})")
    return hashes


def leer_csv(asignatura: str) -> pd.DataFrame:
    """Lee un CSV UCI con tipos explícitos y valida el esquema (regla R-01)."""
    path = UCI_FILES[asignatura]
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_values=[""])
    faltan = set(ESQUEMA_UCI) - set(df.columns)
    sobran = set(df.columns) - set(ESQUEMA_UCI)
    if faltan or sobran:
        raise ValueError(f"R-01 esquema inválido en {path.name}: faltan={faltan} sobran={sobran}")
    # Conversión explícita de tipos: los enteros llegan a veces entrecomillados ("5").
    for col, tipo in ESQUEMA_UCI.items():
        if tipo == "int":
            df[col] = pd.to_numeric(df[col].str.strip().str.strip('"'), errors="raise").astype("int16")
        else:
            df[col] = df[col].astype(str)
    df.insert(0, "asignatura", asignatura)
    df.insert(1, "archivo_origen", path.name)
    df.insert(2, "fila_origen", range(2, len(df) + 2))  # nº de línea en el CSV (1 = cabecera)
    return df


def run() -> pd.DataFrame:
    """Ejecuta la etapa Extract y devuelve el staging crudo concatenado."""
    ensure_dirs()
    descargar_uci()
    hashes = verificar_checksums()
    frames = [leer_csv(a) for a in UCI_FILES]
    raw = pd.concat(frames, ignore_index=True)
    for a, f in zip(UCI_FILES, frames):
        etapa(f"extract_{a}", filas_leidas=len(f), columnas=f.shape[1] - 3, sha256=hashes[a])
    regla("R-01", "F1 UCI", "Esquema: 33 columnas esperadas y tipos declarados",
          "Validar/abortar", 0, "src/extract.py::leer_csv", antes=len(raw), despues=len(raw))
    out = DATA_STAGING / "uci_raw.parquet"
    raw.to_parquet(out, index=False)
    artefacto(out, len(raw))
    etapa("extract", filas_totales=len(raw))
    return raw


if __name__ == "__main__":
    run()
