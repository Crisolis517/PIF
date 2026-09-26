"""TRANSFORM (F2, F3) — limpieza de los JSON simulados y construcción de documentos.

Entradas : data/raw/synthetic/{lms_eventos.jsonl, tutorias.jsonl, alertas.json}
           data/staging/matricula.parquet (integridad referencial contra SQL)
Salidas  : data/staging/sim_seguimiento_docs.json   (colección seguimiento_riesgo)
           data/staging/sim_alertas_docs.json       (colección alertas_riesgo)
           data/staging/sim_cuarentena.json         (registros rechazados + motivo)
           data/staging/sim_eventos_limpios.parquet (eventos planos, para el benchmark)

Reglas R-15 … R-25 (ver catálogo en el informe, sección 5).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime

import pandas as pd

from config import DATA_STAGING, DATA_SYNTH_RAW, PERIODOS
from generate_synthetic import VERSION_GENERADOR
from tracking import artefacto, etapa, regla

IMPL = "src/transform_synthetic.py"
CATALOGO_TIPO = {"recurso", "tarea", "foro", "cuestionario"}
MAPA_TIPO = {"resource": "recurso", "task": "tarea", "forum": "foro", "quiz": "cuestionario"}
CATALOGO_RESULTADO = {"completada", "no_asistio", "reprogramada"}
MAPA_RESULTADO = {"completed": "completada", "no asistio": "no_asistio",
                  "no-asistio": "no_asistio", "rescheduled": "reprogramada"}
ORDEN_ESTADOS = {"abierta": 0, "en_seguimiento": 1, "cerrada": 2}
DUR_MAX = 240  # minutos: sesiones más largas se consideran no cerradas

STOPWORDS = set("""a al con de del el en es la las lo los para por se su sus un una y o que
durante tras sin sobre le les no ha han muy mas""".split())
ABREVIATURAS = {"est.": "estudiante", " q ": " que "}
LEXICO = {  # prefijos (texto normalizado, sin tildes) -> categoría
    "academico": ["rendimiento", "nota", "refuerzo", "ejercicio", "tarea", "recuperacion",
                  "contenido", "minimo"],
    "asistencia": ["inasistencia", "falta", "ausencia", "retraso", "asistencia", "justific"],
    "familiar": ["familia", "hogar", "representante"],
    "conducta": ["conducta", "disruptiv", "conflicto", "companero", "llamado"],
    "motivacion": ["motivacion", "desinteres", "desanimo", "meta", "continuar"],
}


def periodo_de(fecha: datetime | None) -> str | None:
    """R-19: asigna P1/P2/P3 según el calendario lectivo; None si cae fuera."""
    if fecha is None:
        return None
    d = fecha.date()
    for p, v in PERIODOS.items():
        if v["inicio"] <= d <= v["fin"]:
            return p
    return None


def normalizar_texto(texto: str) -> str:
    """R-25: minúsculas, expansión de abreviaturas, sin tildes ni puntuación ni espacios extra."""
    t = f" {texto.lower()} "
    for k, v in ABREVIATURAS.items():
        t = t.replace(k, v)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(texto_norm: str) -> list[str]:
    """Tokens sin palabras vacías."""
    return [w for w in texto_norm.split() if w not in STOPWORDS and len(w) > 2]


def categorizar(texto_norm: str) -> str:
    """R-25: categoría por léxico (máximo de coincidencias; empate -> 'indeterminada')."""
    toks = tokens(texto_norm)
    score = {c: sum(any(t.startswith(p) for p in ps) for t in toks) for c, ps in LEXICO.items()}
    best = max(score.values())
    ganadores = [c for c, s in score.items() if s == best]
    if best == 0 or len(ganadores) > 1:
        return "indeterminada"
    return ganadores[0]


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def limpiar_eventos(eventos: list[dict], matriculas: set, cuarentena: list) -> pd.DataFrame:
    """Aplica R-15 … R-22 a los eventos LMS crudos."""
    n0 = len(eventos)
    req = ["evento_id", "estudiante_id", "asignatura", "fecha", "tipo"]
    ok = []
    for e in eventos:
        if any(e.get(k) in (None, "") for k in req):
            cuarentena.append({"fuente": "lms", "regla": "R-15", "registro": e})
        else:
            ok.append(e)
    regla("R-15", "F2 LMS (sim.)", "Campos obligatorios ausentes (evento_id, estudiante_id, "
          "asignatura, fecha, tipo)", "Cuarentena", n0 - len(ok), f"{IMPL}::limpiar_eventos",
          n0, len(ok))

    n1 = len(ok)
    vistos, dedup = set(), []
    for e in ok:
        if e["evento_id"] not in vistos:
            vistos.add(e["evento_id"])
            dedup.append(e)
    regla("R-16", "F2 LMS (sim.)", "Eventos duplicados (mismo evento_id)",
          "Eliminar, conservar la primera aparición", n1 - len(dedup), f"{IMPL}::limpiar_eventos",
          n1, len(dedup))

    n2 = len(dedup)
    ref = []
    for e in dedup:
        if (e["estudiante_id"], e["asignatura"]) in matriculas:
            ref.append(e)
        else:
            cuarentena.append({"fuente": "lms", "regla": "R-17", "registro": e})
    regla("R-17", "F2 LMS (sim.)", "Integridad referencial: (estudiante_id, asignatura) sin "
          "matrícula en PostgreSQL", "Cuarentena", n2 - len(ref), f"{IMPL}::limpiar_eventos",
          n2, len(ref))

    n3 = len(ref)
    cal = []
    for e in ref:
        f = _parse(e["fecha"])
        p = periodo_de(f)
        if p is None:
            cuarentena.append({"fuente": "lms", "regla": "R-18", "registro": e})
            continue
        cal.append({**e, "fecha": f, "periodo": p})
    regla("R-18", "F2 LMS (sim.)", "Fecha fuera del calendario lectivo 2005–2006",
          "Cuarentena", n3 - len(cal), f"{IMPL}::limpiar_eventos", n3, len(cal))
    regla("R-19", "F2 LMS (sim.)", "Asignación de período (P1/P2/P3) por fecha del evento",
          "Derivar campo periodo", len(cal), f"{IMPL}::periodo_de", len(cal), len(cal))

    df = pd.DataFrame(cal)
    tipo_raw = df.tipo.copy()
    df["tipo"] = df.tipo.str.strip().str.lower().replace(MAPA_TIPO)
    regla("R-20", "F2 LMS (sim.)", "Tipo de evento no estandarizado (mayúsculas, espacios, "
          "inglés)", "Mapear al catálogo {recurso, tarea, foro, cuestionario}",
          int((~tipo_raw.isin(CATALOGO_TIPO)).sum()), f"{IMPL}::limpiar_eventos")
    assert set(df.tipo) <= CATALOGO_TIPO, "R-20: quedan tipos fuera del catálogo"

    neg = df.duracion_min < 0
    df.loc[neg, "duracion_min"] = pd.NA
    regla("R-21", "F2 LMS (sim.)", "Duración negativa (error de registro)",
          "Anular duración (el evento cuenta, no suma minutos)", int(neg.sum()),
          f"{IMPL}::limpiar_eventos")
    exc = df.duracion_min > DUR_MAX
    df.loc[exc, "duracion_min"] = DUR_MAX
    regla("R-22", "F2 LMS (sim.)", f"Duración > {DUR_MAX} min (sesión no cerrada)",
          f"Truncar a {DUR_MAX} min", int(exc.sum()), f"{IMPL}::limpiar_eventos")
    df["duracion_min"] = df.duracion_min.astype("Int64")
    if "revisada" not in df:
        df["revisada"] = pd.NA
    etapa("clean_lms", eventos_crudos=n0, eventos_limpios=len(df))
    return df.sort_values(["estudiante_id", "fecha"]).reset_index(drop=True)


def limpiar_tutorias(tutorias: list[dict], matriculas: set, cuarentena: list) -> pd.DataFrame:
    """R-23: resultado estandarizado, fecha en calendario e integridad referencial."""
    n0 = len(tutorias)
    df = pd.DataFrame(tutorias)
    res_raw = df.resultado.copy()
    df["resultado"] = (df.resultado.str.strip().str.lower()
                       .map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode())
                       .replace(MAPA_RESULTADO))
    n_res = int((~res_raw.isin(CATALOGO_RESULTADO)).sum())
    df["fecha"] = df.fecha.map(_parse)
    df["periodo"] = df.fecha.map(periodo_de)
    malos = df.periodo.isna() | ~df.apply(lambda r: (r.estudiante_id, r.asignatura) in matriculas, axis=1)
    for r in df[malos].to_dict("records"):
        r["fecha"] = r["fecha"].isoformat()
        cuarentena.append({"fuente": "tutorias", "regla": "R-23", "registro": r})
    df = df[~malos].reset_index(drop=True)
    regla("R-23", "F3 Tutorías (sim.)", f"Tutorías: resultado no estandarizado ({n_res} mapeados) "
          f"y fecha fuera de calendario o sin matrícula ({int(malos.sum())} a cuarentena)",
          "Mapear resultado; cuarentena", n_res + int(malos.sum()), f"{IMPL}::limpiar_tutorias",
          n0, len(df))
    assert set(df.resultado) <= CATALOGO_RESULTADO
    etapa("clean_tutorias", tutorias_crudas=n0, tutorias_limpias=len(df))
    return df


def limpiar_alertas(alertas: list[dict], matriculas: set, cuarentena: list) -> list[dict]:
    """R-24 (coherencia temporal del ciclo de vida) y R-25 (texto libre)."""
    corr_act = corr_orden = corr_estado = 0
    limpias = []
    cats = Counter()
    for a in alertas:
        if (a["estudiante_id"], a["asignatura"]) not in matriculas:
            cuarentena.append({"fuente": "alertas", "regla": "R-17", "registro": a})
            continue
        a = json.loads(json.dumps(a))
        for k in ("creada_en", "actualizado_en"):
            a[k] = _parse(a[k])
        for h in a["historial_estados"]:
            h["fecha"] = _parse(h["fecha"])
        for n in a["notas"]:
            n["fecha"] = _parse(n["fecha"])
            n["texto_normalizado"] = normalizar_texto(n["texto"])
            n["categoria"] = categorizar(n["texto_normalizado"])
            cats[n["categoria"]] += 1
        hist = sorted(a["historial_estados"], key=lambda h: (h["fecha"], ORDEN_ESTADOS[h["estado"]]))
        if [h["estado"] for h in hist] != [h["estado"] for h in a["historial_estados"]]:
            corr_orden += 1
        a["historial_estados"] = hist
        if a["estado"] != hist[-1]["estado"]:
            a["estado"] = hist[-1]["estado"]
            corr_estado += 1
        ultima = max([a["creada_en"]] + [h["fecha"] for h in hist] + [n["fecha"] for n in a["notas"]])
        if a["actualizado_en"] < ultima:
            a["actualizado_en"] = ultima
            corr_act += 1
        a["es_simulado"] = True
        a["fuente"] = VERSION_GENERADOR
        limpias.append(a)
    regla("R-24", "F3 Alertas (sim.)", "Coherencia temporal: actualizado_en >= última fecha del "
          f"historial/notas (corregidas {corr_act}); historial ordenado ({corr_orden}); estado = "
          f"último del historial ({corr_estado})", "Corregir y registrar",
          corr_act + corr_orden + corr_estado, f"{IMPL}::limpiar_alertas", len(alertas), len(limpias))
    n_notas = sum(len(a["notas"]) for a in limpias)
    regla("R-25", "F3 Alertas (sim.)", "Texto libre: normalización (minúsculas, tildes, "
          "abreviaturas, puntuación) y categorización por léxico "
          f"({dict(sorted(cats.items()))})", "Derivar texto_normalizado y categoria", n_notas,
          f"{IMPL}::normalizar_texto / categorizar")
    etapa("clean_alertas", alertas=len(limpias), notas=n_notas, **{f"cat_{k}": v for k, v in cats.items()})
    return limpias


def construir_seguimiento(est_ids: list[int], ev: pd.DataFrame, tu: pd.DataFrame) -> list[dict]:
    """Un documento por estudiante × período (unidad de la colección seguimiento_riesgo)."""
    docs = []
    ev_g = {k: g for k, g in ev.groupby(["estudiante_id", "periodo"])}
    tu_g = {k: g for k, g in tu.groupby(["estudiante_id", "periodo"])}
    for e in est_ids:
        for p, v in PERIODOS.items():
            g = ev_g.get((e, p))
            inter = []
            if g is not None:
                for r in g.itertuples():
                    d = {"evento_id": r.evento_id, "asignatura": r.asignatura,
                         "fecha": r.fecha.to_pydatetime(), "tipo": r.tipo,
                         "duracion_min": None if pd.isna(r.duracion_min) else int(r.duracion_min)}
                    if r.tipo == "tarea" and not pd.isna(r.revisada):
                        d["revisada"] = bool(r.revisada)
                    inter.append(d)
            t = tu_g.get((e, p))
            tuts = [] if t is None else [
                {"tutoria_id": r.tutoria_id, "asignatura": r.asignatura,
                 "fecha": r.fecha.to_pydatetime(), "tutor": r.tutor, "modalidad": r.modalidad,
                 "tema": r.tema, "resultado": r.resultado} for r in t.sort_values("fecha").itertuples()]
            docs.append({
                "estudiante_id": int(e), "periodo": p,
                "periodo_inicio": datetime.combine(v["inicio"], datetime.min.time()),
                "periodo_fin": datetime.combine(v["fin"], datetime.max.time().replace(microsecond=0)),
                "interacciones_lms": inter, "tutorias": tuts,
                "es_simulado": True, "fuente": VERSION_GENERADOR,
            })
    return docs


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=lambda o: {"$date": o.isoformat()}
                  if isinstance(o, datetime) else str(o))


def run() -> dict:
    """Ejecuta la limpieza de F2/F3 y deja los documentos listos para MongoDB."""
    mat = pd.read_parquet(DATA_STAGING / "matricula.parquet")
    est = pd.read_parquet(DATA_STAGING / "estudiante.parquet")
    cod = {1: "MAT", 2: "POR"}
    matriculas = set(zip(mat.estudiante_id.astype(int), mat.asignatura_id.map(cod)))
    with open(DATA_SYNTH_RAW / "lms_eventos.jsonl", encoding="utf-8") as f:
        eventos = [json.loads(line) for line in f]
    with open(DATA_SYNTH_RAW / "tutorias.jsonl", encoding="utf-8") as f:
        tutorias = [json.loads(line) for line in f]
    with open(DATA_SYNTH_RAW / "alertas.json", encoding="utf-8") as f:
        alertas = json.load(f)

    cuarentena: list = []
    ev = limpiar_eventos(eventos, matriculas, cuarentena)
    tu = limpiar_tutorias(tutorias, matriculas, cuarentena)
    al = limpiar_alertas(alertas, matriculas, cuarentena)
    docs = construir_seguimiento(sorted(est.estudiante_id.astype(int)), ev, tu)

    ev.to_parquet(DATA_STAGING / "sim_eventos_limpios.parquet", index=False)
    tu.to_parquet(DATA_STAGING / "sim_tutorias_limpias.parquet", index=False)
    _dump(docs, DATA_STAGING / "sim_seguimiento_docs.json")
    _dump(al, DATA_STAGING / "sim_alertas_docs.json")
    _dump(cuarentena, DATA_STAGING / "sim_cuarentena.json")
    for n in ("sim_eventos_limpios.parquet", "sim_seguimiento_docs.json", "sim_alertas_docs.json",
              "sim_cuarentena.json"):
        artefacto(DATA_STAGING / n)
    etapa("transform_synthetic", docs_seguimiento=len(docs), docs_alertas=len(al),
          registros_cuarentena=len(cuarentena),
          cuarentena_por_regla=dict(Counter(c["regla"] for c in cuarentena)))
    return {"seguimiento": docs, "alertas": al, "cuarentena": cuarentena, "eventos": ev}


if __name__ == "__main__":
    run()
