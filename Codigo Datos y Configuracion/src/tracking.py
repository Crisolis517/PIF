"""Trazabilidad de la ejecución: conteos por etapa, reglas de calidad y checksums.

Cada etapa del pipeline registra aquí lo que leyó, rechazó y cargó. Al final se
escriben `logs/ultima_ejecucion.json`, `logs/reglas_calidad.csv` y un log de texto
con marca temporal, de modo que cada número del informe tiene un origen verificable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import LOGS, ROOT

_STATE: dict[str, Any] = {"etapas": {}, "reglas": [], "artefactos": {}, "inicio": None}
_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Devuelve el logger del pipeline (consola + archivo con marca temporal)."""
    global _LOGGER
    if _LOGGER is None:
        LOGS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOGGER = logging.getLogger("pipeline")
        _LOGGER.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(LOGS / f"pipeline_{stamp}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _LOGGER.addHandler(fh)
        _LOGGER.addHandler(sh)
        _STATE["inicio"] = datetime.now().isoformat(timespec="seconds")
    return _LOGGER


def sha256(path: Path) -> str:
    """SHA-256 de un archivo (para verificar integridad de entradas y salidas)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def etapa(nombre: str, **conteos: Any) -> None:
    """Registra conteos de una etapa (p. ej. filas_leidas, rechazadas, cargadas)."""
    _STATE["etapas"].setdefault(nombre, {}).update(conteos)
    get_logger().info("[%s] %s", nombre, json.dumps(conteos, ensure_ascii=False, default=str))


def regla(rid: str, fuente: str, descripcion: str, accion: str, afectados: int,
          implementacion: str, antes: int | None = None, despues: int | None = None) -> None:
    """Registra la aplicación de una regla de calidad con su conteo de registros afectados."""
    _STATE["reglas"].append({
        "id": rid, "fuente": fuente, "descripcion": descripcion, "accion": accion,
        "afectados": int(afectados), "antes": antes, "despues": despues,
        "implementacion": implementacion,
    })
    get_logger().info("[%s] %s -> afectados=%s", rid, descripcion, afectados)


def artefacto(path: Path, filas: int | None = None) -> None:
    """Registra un artefacto de salida con su checksum."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    _STATE["artefactos"][rel] = {"sha256": sha256(path), "bytes": Path(path).stat().st_size,
                                 "filas": filas}


class Cronometro:
    """Context manager que mide la duración de una etapa y la registra."""

    def __init__(self, nombre: str):
        self.nombre = nombre

    def __enter__(self):
        self.t0 = time.perf_counter()
        get_logger().info("==> Inicio etapa: %s", self.nombre)
        return self

    def __exit__(self, *exc):
        dur = round(time.perf_counter() - self.t0, 3)
        etapa(self.nombre, duracion_s=dur)
        return False


def guardar() -> None:
    """Persiste el estado de la ejecución en logs/."""
    LOGS.mkdir(parents=True, exist_ok=True)
    _STATE["fin"] = datetime.now().isoformat(timespec="seconds")
    with open(LOGS / "ultima_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(_STATE, f, ensure_ascii=False, indent=2, default=str)
    if _STATE["reglas"]:
        pd.DataFrame(_STATE["reglas"]).to_csv(LOGS / "reglas_calidad.csv", index=False)
