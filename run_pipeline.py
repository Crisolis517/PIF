"""Punto de entrada único: reconstruye TODO desde los CSV originales de UCI.

    python run_pipeline.py              # pipeline completo + tablas del informe
    python run_pipeline.py --informe    # además compila el PDF (requiere tectonic)
    python run_pipeline.py --sin-benchmark --sin-notebook   # versión rápida

Etapas: extract -> transform (F1) -> generate_synthetic (F2/F3) -> transform_synthetic
-> load (PostgreSQL + MongoDB) -> integrate -> analysis -> benchmark_escalabilidad
-> notebook (ejecutado) -> report_tables. Cada etapa registra conteos en logs/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import tracking  # noqa: E402
from tracking import Cronometro, get_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-benchmark", action="store_true", help="omite el benchmark de escalabilidad")
    ap.add_argument("--sin-notebook", action="store_true", help="omite la ejecución del notebook")
    ap.add_argument("--informe", action="store_true", help="compila informe/main.tex con tectonic")
    args = ap.parse_args()
    log = get_logger()

    import extract, transform, generate_synthetic, transform_synthetic, load, integrate, analysis  # noqa: E401
    etapas = [("1_extract", extract.run), ("2_transform_uci", transform.run),
              ("3_generate_synthetic", generate_synthetic.run),
              ("4_transform_synthetic", transform_synthetic.run),
              ("5_load_postgres", load.cargar_postgres), ("6_load_mongo", load.cargar_mongo),
              ("7_integrate", integrate.run), ("8_analysis", analysis.run)]
    if not args.sin_benchmark:
        import benchmark_escalabilidad
        etapas.append(("9_benchmark", benchmark_escalabilidad.run))
    try:
        for nombre, fn in etapas:
            with Cronometro(nombre):
                fn()
        if not args.sin_notebook:
            with Cronometro("10_notebook"):
                subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                                "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
                                str(ROOT / "notebooks" / "analisis.ipynb")], check=True)
    finally:
        tracking.guardar()
    import report_tables
    with Cronometro("11_report_tables"):
        report_tables.run()
    tracking.guardar()
    if args.informe:
        tectonic = shutil.which("tectonic") or str(ROOT / "tools" / "tectonic")
        subprocess.run([tectonic, "main.tex"], cwd=ROOT / "informe", check=True)
        shutil.copy(ROOT / "informe" / "main.pdf", ROOT / "informe" / "Proyecto_Integrador_Final_Grupo1.pdf")
    log.info("Pipeline completo. Resumen en logs/ultima_ejecucion.json")


if __name__ == "__main__":
    main()
