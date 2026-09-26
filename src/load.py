"""LOAD — carga en PostgreSQL (F1) y MongoDB (F2, F3) con verificaciones posteriores.

PostgreSQL: ejecuta sql/01_ddl.sql, inserta las seis tablas en orden de dependencias
dentro de una transacción, crea vistas/roles (02) y ejecuta los controles de
integridad (03) y las consultas analíticas (04). Salidas en results/sql_*.csv.

MongoDB: crea las colecciones con validador ($jsonSchema + $expr, validationAction =
error), índices justificados (mongo/indices.json), inserta los documentos limpios,
prueba que el validador rechaza un documento incoherente y ejecuta los pipelines
de mongo/pipelines.json. Salidas en results/mongo_*.csv.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, WriteError
from sqlalchemy import create_engine, text

from config import (DATA_STAGING, MONGO_DB, MONGO_DIR, MONGO_URI, RESULTS, SQL_DIR, ensure_dirs,
                    pg_url)
from tracking import etapa, get_logger

ORDEN_TABLAS = ["escuela", "asignatura", "periodo", "estudiante", "matricula", "calificacion"]


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #
def consultas_nombradas(path) -> list[tuple[str, str]]:
    """Divide un .sql en bloques '-- name: ID descripción' -> (id_desc, sql)."""
    txt = path.read_text(encoding="utf-8")
    partes = re.split(r"^-- name: (.+)$", txt, flags=re.M)
    return [(partes[i].strip(), partes[i + 1].strip()) for i in range(1, len(partes), 2)]


def cargar_postgres() -> dict:
    """Crea el esquema, carga las tablas y ejecuta integridad + analíticas."""
    log = get_logger()
    eng = create_engine(pg_url())
    tablas = {t: pd.read_parquet(DATA_STAGING / f"{t}.parquet") for t in ORDEN_TABLAS}
    with eng.begin() as con:
        con.exec_driver_sql((SQL_DIR / "01_ddl.sql").read_text(encoding="utf-8"))
        for t in ORDEN_TABLAS:
            tablas[t].to_sql(t, con, schema="academico", if_exists="append", index=False,
                             method="multi", chunksize=1000)
        con.exec_driver_sql((SQL_DIR / "02_vistas_seguridad.sql").read_text(encoding="utf-8"))
        con.exec_driver_sql("ANALYZE;")
        version = con.execute(text("SHOW server_version")).scalar()
    conteos = {}
    with eng.connect() as con:
        for t in ORDEN_TABLAS:
            conteos[t] = con.execute(text(f"SELECT COUNT(*) FROM academico.{t}")).scalar()
    etapa("load_postgres", servidor=f"PostgreSQL {version}", **{f"filas_{k}": v for k, v in conteos.items()},
          total_registros=sum(conteos.values()))

    # Integridad
    filas = []
    with eng.connect() as con:
        for nombre, q in consultas_nombradas(SQL_DIR / "03_integridad.sql"):
            r = con.execute(text(q)).mappings().one()
            filas.append({**r, "ok": r["valor"] == r["esperado"], "descripcion": nombre})
    integ = pd.DataFrame(filas)
    integ.to_csv(RESULTS / "sql_integridad.csv", index=False)
    fallos = integ[~integ.ok]
    etapa("sql_integridad", controles=len(integ), superados=int(integ.ok.sum()))
    if len(fallos):
        raise AssertionError(f"Controles de integridad fallidos:\n{fallos}")

    # Analíticas
    with eng.connect() as con:
        for nombre, q in consultas_nombradas(SQL_DIR / "04_analiticas.sql"):
            qid = nombre.split()[0]
            df = pd.read_sql(text(q), con)
            df.to_csv(RESULTS / f"sql_{qid}.csv", index=False)
            log.info("Consulta %s -> %d filas", qid, len(df))
        plan = "\n".join(r[0] for r in con.execute(text(
            "EXPLAIN SELECT a.codigo, p.codigo, AVG(c.nota) FROM academico.calificacion c "
            "JOIN academico.asignatura a USING (asignatura_id) JOIN academico.periodo p USING (periodo_id) "
            "WHERE NOT c.no_evaluado GROUP BY 1, 2")))
    (RESULTS / "sql_explain_A01.txt").write_text(plan, encoding="utf-8")
    return conteos


# --------------------------------------------------------------------------- #
# MongoDB
# --------------------------------------------------------------------------- #
def _hook(d):
    if set(d) == {"$date"}:
        return datetime.fromisoformat(d["$date"])
    return d


def _leer(nombre):
    with open(DATA_STAGING / nombre, encoding="utf-8") as f:
        return json.load(f, object_hook=_hook)


def _limpiar_para_csv(docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(docs)


def cargar_mongo() -> dict:
    """Crea colecciones validadas, índices, inserta documentos y ejecuta pipelines."""
    log = get_logger()
    cli = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    version = cli.server_info()["version"]
    db = cli[MONGO_DB]
    validadores = json.loads((MONGO_DIR / "validadores.json").read_text(encoding="utf-8"))
    indices = json.loads((MONGO_DIR / "indices.json").read_text(encoding="utf-8"))
    for col in ("seguimiento_riesgo", "alertas_riesgo", "cuarentena"):
        db.drop_collection(col)
    for col, val in validadores.items():
        db.create_collection(col, validator=val, validationLevel="strict", validationAction="error")
        for ix in indices[col]:
            kw = {k: v for k, v in ix.items() if k in ("unique", "name", "default_language")}
            db[col].create_index([tuple(k) for k in ix["keys"]], **kw)
    db.create_collection("cuarentena")

    seg = _leer("sim_seguimiento_docs.json")
    ale = _leer("sim_alertas_docs.json")
    cua = _leer("sim_cuarentena.json")
    rechazados = 0
    for col, docs in (("seguimiento_riesgo", seg), ("alertas_riesgo", ale)):
        try:
            db[col].insert_many(docs, ordered=False)
        except BulkWriteError as e:  # documentos que no cumplen el validador
            rechazados += len(e.details["writeErrors"])
            for w in e.details["writeErrors"]:
                db.cuarentena.insert_one({"fuente": col, "regla": "validador", "error": w["errmsg"][:500]})
    if cua:
        db.cuarentena.insert_many([{**c, "registro": json.loads(json.dumps(c["registro"], default=str))}
                                   for c in cua])

    # Prueba negativa del validador: reproduce el defecto del avance
    # (actualizado_en anterior a la fecha de una nota) y verifica el rechazo.
    malo = json.loads(json.dumps(ale[0], default=str), object_hook=_hook)
    for k in ("creada_en", "actualizado_en"):
        malo[k] = datetime.fromisoformat(malo[k]) if isinstance(malo[k], str) else malo[k]
    for h in malo["historial_estados"]:
        h["fecha"] = datetime.fromisoformat(h["fecha"]) if isinstance(h["fecha"], str) else h["fecha"]
    for n in malo["notas"]:
        n["fecha"] = datetime.fromisoformat(n["fecha"]) if isinstance(n["fecha"], str) else n["fecha"]
    malo["alerta_id"] = "AL-99999"
    malo["actualizado_en"] = malo["creada_en"] - timedelta(days=5)
    try:
        db.alertas_riesgo.insert_one(malo)
        rechazo_ok = False
        db.alertas_riesgo.delete_one({"alerta_id": "AL-99999"})
    except WriteError:
        rechazo_ok = True
    malo2 = {"estudiante_id": 1, "periodo": "P1", "periodo_inicio": datetime(2005, 9, 15),
             "periodo_fin": datetime(2005, 12, 16, 23, 59, 59), "es_simulado": True, "fuente": "prueba",
             "tutorias": [], "interacciones_lms": [{"evento_id": "EV-9999999", "asignatura": "MAT",
                                                    "fecha": datetime(2006, 7, 1), "tipo": "foro",
                                                    "duracion_min": 10}]}
    try:
        db.seguimiento_riesgo.insert_one({**malo2, "estudiante_id": 99999})
        rechazo2_ok = False
    except WriteError:
        rechazo2_ok = True

    conteos = {
        "seguimiento_riesgo": db.seguimiento_riesgo.count_documents({}),
        "alertas_riesgo": db.alertas_riesgo.count_documents({}),
        "cuarentena": db.cuarentena.count_documents({}),
        "eventos_lms_embebidos": next(db.seguimiento_riesgo.aggregate(
            [{"$group": {"_id": None, "n": {"$sum": {"$size": "$interacciones_lms"}}}}]))["n"],
        "tutorias_embebidas": next(db.seguimiento_riesgo.aggregate(
            [{"$group": {"_id": None, "n": {"$sum": {"$size": "$tutorias"}}}}]))["n"],
        "estudiantes_cubiertos": len(db.seguimiento_riesgo.distinct("estudiante_id")),
    }
    etapa("load_mongo", servidor=f"MongoDB {version}", rechazados_validador=rechazados,
          prueba_validador_alerta_incoherente_rechazada=rechazo_ok,
          prueba_validador_evento_fuera_periodo_rechazado=rechazo2_ok, **conteos)
    if not (rechazo_ok and rechazo2_ok):
        raise AssertionError("El validador no rechazó un documento incoherente")

    # Pipelines documentados: se guardan las salidas de los resúmenes
    pipes = json.loads((MONGO_DIR / "pipelines.json").read_text(encoding="utf-8"))
    for pid, spec in pipes.items():
        res = list(db[spec["coleccion"]].aggregate(spec["pipeline"]))
        pd.DataFrame(res).to_csv(RESULTS / f"mongo_{pid.split('_')[0]}.csv", index=False)
        log.info("Pipeline %s -> %d documentos", pid, len(res))
    ejemplo = db.seguimiento_riesgo.find_one({"estudiante_id": 1, "periodo": "P1"}, {"_id": 0})
    ejemplo["interacciones_lms"] = ejemplo["interacciones_lms"][:2]
    ej_al = db.alertas_riesgo.find_one({"notas.1": {"$exists": True}}, {"_id": 0})
    with open(RESULTS / "mongo_documentos_ejemplo.json", "w", encoding="utf-8") as f:
        json.dump({"seguimiento_riesgo": ejemplo, "alertas_riesgo": ej_al}, f, ensure_ascii=False,
                  indent=2, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))
    stats = {c: db.command("collstats", c) for c in ("seguimiento_riesgo", "alertas_riesgo")}
    etapa("mongo_almacenamiento", **{f"{c}_bytes": int(s["size"]) for c, s in stats.items()},
          **{f"{c}_indices_bytes": int(s["totalIndexSize"]) for c, s in stats.items()})
    return conteos


def run() -> None:
    """Ejecuta la etapa Load completa."""
    ensure_dirs()
    cargar_postgres()
    cargar_mongo()


if __name__ == "__main__":
    run()
