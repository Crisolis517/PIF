"""ESCALABILIDAD — benchmark medido para decidir si el caso requiere Spark.

Replica k veces los eventos LMS limpios (con estudiantes nuevos, para que también
escalen los grupos) y mide, en un solo nodo con pandas + pyarrow:
  - huella en memoria del DataFrame (bytes/fila),
  - tamaño en disco en Parquet (snappy) y tiempos de escritura/lectura,
  - tiempo de la agregación equivalente al pipeline AG-01 (groupby por
    estudiante × asignatura × período: conteo, suma de minutos, días distintos).
También mide el pipeline AG-01 real en MongoDB (escala ×1).
Salidas: results/escalabilidad.csv, results/escalabilidad_resumen.json,
         informe/figuras/fig_escalabilidad.png
"""
from __future__ import annotations

import json
import os
import platform
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from config import DATA_STAGING, FIGURES, MONGO_DB, MONGO_DIR, MONGO_URI, RESULTS, ensure_dirs  # noqa: E402
from tracking import etapa, get_logger  # noqa: E402

FACTORES = [1, 10, 100, 250, 500]
LIMITE_BYTES = 2.0e9  # no superar ~2 GB de huella en el equipo de pruebas (8 GB RAM)


def _base() -> pd.DataFrame:
    ev = pd.read_parquet(DATA_STAGING / "sim_eventos_limpios.parquet")
    return pd.DataFrame({
        "estudiante_id": ev.estudiante_id.astype("int32"),
        "asignatura": ev.asignatura.astype("category"),
        "periodo": ev.periodo.astype("category"),
        "fecha": pd.to_datetime(ev.fecha),
        "tipo": ev.tipo.astype("category"),
        "duracion_min": ev.duracion_min.astype("float32"),
    })


def _escalar(base: pd.DataFrame, k: int) -> pd.DataFrame:
    if k == 1:
        return base
    off = np.repeat(np.arange(k, dtype="int32") * 1000, len(base))
    df = pd.concat([base] * k, ignore_index=True)
    df["estudiante_id"] = df.estudiante_id.values + off
    return df


def _agregar(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalente pandas de AG-01."""
    g = df.assign(dia=df.fecha.dt.normalize()).groupby(
        ["estudiante_id", "asignatura", "periodo"], observed=True)
    return g.agg(n_eventos=("tipo", "size"), minutos=("duracion_min", "sum"), dias=("dia", "nunique"))


def run() -> pd.DataFrame:
    ensure_dirs()
    log = get_logger()
    base = _base()
    filas = []
    tmp = tempfile.mkdtemp()
    for k in FACTORES:
        df = _escalar(base, k)
        mem = int(df.memory_usage(deep=True).sum())
        if mem > LIMITE_BYTES:
            log.info("Escala x%d omitida: %.2f GB > límite", k, mem / 1e9)
            break
        t0 = time.perf_counter(); agg = _agregar(df); t_agg = time.perf_counter() - t0
        path = os.path.join(tmp, f"ev_{k}.parquet")
        t0 = time.perf_counter(); df.to_parquet(path, index=False); t_w = time.perf_counter() - t0
        t0 = time.perf_counter(); pd.read_parquet(path); t_r = time.perf_counter() - t0
        filas.append({"factor": k, "filas": len(df), "grupos": len(agg), "memoria_bytes": mem,
                      "bytes_por_fila": mem / len(df), "parquet_bytes": os.path.getsize(path),
                      "t_agregacion_s": t_agg, "t_escritura_s": t_w, "t_lectura_s": t_r,
                      "filas_por_s_agregacion": len(df) / t_agg})
        log.info("Escala x%d: %d filas, agregación %.2f s, %.0f MB", k, len(df), t_agg, mem / 1e6)
        os.remove(path)
        del df, agg
    res = pd.DataFrame(filas)
    res.to_csv(RESULTS / "escalabilidad.csv", index=False)

    # MongoDB, escala real
    db = MongoClient(MONGO_URI)[MONGO_DB]
    spec = json.loads((MONGO_DIR / "pipelines.json").read_text(encoding="utf-8"))["AG-01_lms_por_matricula_periodo"]
    tiempos = []
    for _ in range(5):
        t0 = time.perf_counter(); list(db.seguimiento_riesgo.aggregate(spec["pipeline"])); tiempos.append(time.perf_counter() - t0)

    # Proyección (supuestos explícitos): eventos/matrícula/año medidos en el generador
    ev_por_matricula = len(base) / 1044
    bpf = float(res.bytes_por_fila.iloc[-1])
    rps = float(res.filas_por_s_agregacion.iloc[-1])
    ram = 8e9
    filas_max_nodo = ram / (3 * bpf)  # margen x3 para intermedios del groupby
    escenarios = []
    for nombre, matriculas, mult in [("Caso actual (UCI)", 1044, 1),
                                     ("Red de 100 centros (100 000 est. × 10 asignaturas)", 1_000_000, 1),
                                     ("Red nacional (1 000 000 est. × 10 asignaturas)", 10_000_000, 1),
                                     ("Red nacional con clickstream (×20 eventos)", 10_000_000, 20)]:
        n = matriculas * ev_por_matricula * mult
        escenarios.append({"escenario": nombre, "eventos_anuales": n, "memoria_gb": n * bpf / 1e9,
                           "t_agregacion_estimado_s": n / rps, "cabe_en_un_nodo_8gb": n < filas_max_nodo})
    esc = pd.DataFrame(escenarios)
    esc.to_csv(RESULTS / "escalabilidad_escenarios.csv", index=False)
    resumen = {"equipo": f"{platform.machine()} · {os.cpu_count()} núcleos · 8 GB RAM · {platform.system()}",
               "eventos_por_matricula_anio": ev_por_matricula, "bytes_por_fila": bpf,
               "filas_por_s": rps, "filas_max_nodo_8gb": filas_max_nodo,
               "mongo_ag01_mediana_s": float(np.median(tiempos)),
               "dataset_uci_bytes": int(sum(os.path.getsize(p) for p in
                                            [DATA_STAGING.parent / "raw" / "student-mat.csv",
                                             DATA_STAGING.parent / "raw" / "student-por.csv"]))}
    with open(RESULTS / "escalabilidad_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axs[0].loglog(res.filas, res.t_agregacion_s, marker="o", color="#2a78d6", label="agregación AG-01")
    axs[0].loglog(res.filas, res.t_lectura_s, marker="o", color="#eb6834", label="lectura Parquet")
    axs[0].set(xlabel="Eventos LMS (filas)", ylabel="Tiempo (s)", title="Tiempo en un nodo (pandas)")
    axs[0].legend(fontsize=7)
    axs[1].loglog(res.filas, res.memoria_bytes / 1e6, marker="o", color="#2a78d6", label="memoria (MB)")
    axs[1].loglog(res.filas, res.parquet_bytes / 1e6, marker="o", color="#1baf7a", label="Parquet en disco (MB)")
    axs[1].axhline(ram / 3 / 1e6, color="#8a8985", ls="--", lw=1)
    axs[1].text(res.filas.iloc[0], ram / 3 / 1e6 * 1.2, "RAM/3 del nodo", fontsize=7, color="#8a8985")
    axs[1].set(xlabel="Eventos LMS (filas)", ylabel="MB", title="Huella de datos")
    axs[1].legend(fontsize=7, loc="lower right")
    for ax in axs:
        ax.title.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_escalabilidad.png")
    plt.close(fig)
    etapa("benchmark_escalabilidad", escalas=len(res), filas_max=int(res.filas.max()),
          t_agregacion_max_s=round(float(res.t_agregacion_s.max()), 3))
    return res


if __name__ == "__main__":
    run()
