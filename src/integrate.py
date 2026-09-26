"""INTEGRATE — dataset analítico SQL + MongoDB.

Unidad de análisis: la MATRÍCULA (estudiante × asignatura), observada con la
información disponible en cada corte temporal. Una fila por matrícula: 1.044 filas.
Las notas de cada período (P1<->G1, P2<->G2, P3<->G3) son columnas; las variables
simuladas llevan el prefijo `sim_`.

Fuentes: vista academico.v_matricula_ancha (PostgreSQL) + pipelines AG-01, AG-02 y
AG-03 (MongoDB). Regla de integración: LEFT JOIN desde la matrícula por
(estudiante_id, asignatura); ausencia de actividad simulada = 0 (R-26).

Salidas: data/processed/dataset_analitico.parquet (+ .csv),
         data/processed/diccionario_datos.csv, data/processed/linaje.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import StratifiedGroupKFold
from sqlalchemy import create_engine, text

from config import (DATA_PROCESSED, MONGO_DB, MONGO_DIR, MONGO_URI, SEED, UMBRAL_APROBACION,
                    ensure_dirs, pg_url)
from tracking import artefacto, etapa, regla

IMPL = "src/integrate.py"

# nombre, tipo, dominio, origen, fuente/campo, transformación (regla), descripción
DICCIONARIO = [
    ("estudiante_id", "int32", "1–674", "F1 real", "academico.matricula.estudiante_id", "R-12 identidad", "Identificador seudonimizado del estudiante"),
    ("asignatura", "category", "MAT, POR", "F1 real", "academico.asignatura.codigo", "R-06", "Asignatura de la matrícula"),
    ("escuela", "category", "GP, MS", "F1 real", "academico.escuela.codigo (school)", "R-06", "Centro educativo"),
    ("sexo", "category", "F, M", "F1 real", "estudiante.sexo (sex)", "R-06", "Sexo declarado"),
    ("edad", "int8", "15–22", "F1 real", "estudiante.edad (age)", "—", "Edad en años"),
    ("zona", "category", "Urbana, Rural", "F1 real", "estudiante.zona (address)", "R-06", "Tipo de domicilio"),
    ("tamano_familia", "category", "<=3, >3", "F1 real", "estudiante.tamano_familia (famsize)", "R-06", "Tamaño del hogar"),
    ("estado_padres", "category", "Juntos, Separados", "F1 real", "estudiante.estado_padres (Pstatus)", "R-06", "Convivencia de los padres"),
    ("educ_madre", "int8", "0–4", "F1 real", "estudiante.educ_madre (Medu)", "—", "Nivel educativo de la madre"),
    ("educ_padre", "int8", "0–4", "F1 real", "estudiante.educ_padre (Fedu)", "—", "Nivel educativo del padre"),
    ("trabajo_madre", "category", "5 niveles", "F1 real", "estudiante.trabajo_madre (Mjob)", "R-06", "Ocupación de la madre"),
    ("trabajo_padre", "category", "5 niveles", "F1 real", "estudiante.trabajo_padre (Fjob)", "R-06", "Ocupación del padre"),
    ("motivo_eleccion", "category", "4 niveles", "F1 real", "estudiante.motivo_eleccion (reason)", "R-06", "Motivo de elección del centro"),
    ("tutor_legal", "category", "madre, padre, otro", "F1 real", "estudiante.tutor_legal (guardian)", "R-06", "Tutor legal"),
    ("tiempo_viaje", "int8", "1–4", "F1 real", "estudiante.tiempo_viaje (traveltime)", "—", "Tiempo de traslado (ordinal)"),
    ("tiempo_estudio", "int8", "1–4", "F1 real", "estudiante.tiempo_estudio (studytime)", "—", "Estudio semanal: 1 <2 h, 2 2–5 h, 3 5–10 h, 4 >10 h"),
    ("apoyo_escolar", "bool", "V/F", "F1 real", "estudiante.apoyo_escolar (schoolsup)", "R-06", "Refuerzo educativo del centro"),
    ("apoyo_familiar", "bool", "V/F", "F1 real", "estudiante.apoyo_familiar (famsup)", "R-06", "Apoyo educativo familiar"),
    ("actividades", "bool", "V/F", "F1 real", "estudiante.actividades (activities)", "R-06", "Actividades extracurriculares"),
    ("guarderia", "bool", "V/F", "F1 real", "estudiante.guarderia (nursery)", "R-06", "Asistió a educación infantil"),
    ("desea_superior", "bool", "V/F", "F1 real", "estudiante.desea_superior (higher)", "R-06", "Aspira a educación superior"),
    ("internet", "bool", "V/F", "F1 real", "estudiante.internet (internet)", "R-06", "Internet en el hogar"),
    ("relacion_romantica", "bool", "V/F", "F1 real", "estudiante.relacion_romantica (romantic)", "R-06", "Relación sentimental"),
    ("relacion_familiar", "int8", "1–5", "F1 real", "estudiante.relacion_familiar (famrel)", "—", "Calidad de relaciones familiares"),
    ("tiempo_libre", "int8", "1–5", "F1 real", "estudiante.tiempo_libre (freetime)", "—", "Tiempo libre"),
    ("salidas", "int8", "1–5", "F1 real", "estudiante.salidas (goout)", "—", "Salidas con amigos"),
    ("alcohol_semana", "int8", "1–5", "F1 real", "estudiante.alcohol_semana (Dalc)", "—", "Consumo de alcohol entre semana"),
    ("alcohol_finde", "int8", "1–5", "F1 real", "estudiante.alcohol_finde (Walc)", "—", "Consumo de alcohol en fin de semana"),
    ("salud", "int8", "1–5", "F1 real", "estudiante.salud (health)", "—", "Estado de salud"),
    ("reprobaciones_previas", "int8", "0–3 (3 = 3+)", "F1 real", "matricula.reprobaciones_previas (failures)", "R-07", "Reprobaciones previas en la asignatura"),
    ("reprobaciones_3mas", "bool", "V/F", "Derivada", "reprobaciones_previas = 3", "R-07", "Marca de valor truncado (3 o más)"),
    ("clases_pagadas", "bool", "V/F", "F1 real", "matricula.clases_pagadas (paid)", "R-06", "Clases particulares pagadas en la asignatura"),
    ("faltas", "int16", "0–75", "F1 real", "matricula.faltas (absences)", "—", "Faltas ANUALES (no disponibles al corte P1; ver L-02)"),
    ("faltas_confiable", "bool", "V/F", "Derivada", "matricula.faltas_confiable", "R-09", "Falso si G3 no evaluado y 0 faltas"),
    ("faltas_atipicas", "bool", "V/F", "Derivada", "faltas > Q3 + 3·RIC por asignatura", "R-10", "Faltas extremas (se conservan)"),
    ("nota_p1", "int8", "0–20", "F1 real", "calificacion.nota, periodo P1 (G1)", "R-14", "Nota del 1.er período"),
    ("nota_p2", "int8", "0–20", "F1 real", "calificacion.nota, periodo P2 (G2)", "R-14", "Nota del 2.º período"),
    ("nota_p3", "int8", "0–20", "F1 real", "calificacion.nota, periodo P3 (G3)", "R-14", "Nota final (solo para construir la variable objetivo)"),
    ("no_evaluado_p1", "bool", "V/F", "Derivada", "calificacion.no_evaluado, P1", "R-08", "Nota 0 en P1 = no evaluado"),
    ("no_evaluado_p2", "bool", "V/F", "Derivada", "calificacion.no_evaluado, P2", "R-08", "Nota 0 en P2 = no evaluado"),
    ("no_evaluado_p3", "bool", "V/F", "Derivada", "calificacion.no_evaluado, P3", "R-08", "Nota 0 en P3 = no evaluado / abandono"),
    ("en_ambas_asignaturas", "bool", "V/F", "Derivada", "COUNT(matricula) = 2", "R-12", "El estudiante cursa MAT y POR"),
    ("en_riesgo", "int8", "0/1", "Derivada (objetivo)", "nota_p3 < 10", "Def. operativa", "1 = reprueba la asignatura (G3 < 10, incluye no evaluados)"),
    ("rendimiento_sobre_promedio", "boolean", "V/F/NA", "Derivada", "nota_p3 > media de la asignatura", "P2", "Solo evaluados en P3; NA si no evaluado"),
    ("sim_lms_eventos_p1", "int32", ">= 0", "F2 SIMULADO", "seguimiento_riesgo.interacciones_lms (P1), AG-01", "R-15…R-22, R-26", "Eventos LMS en P1"),
    ("sim_lms_minutos_p1", "int32", ">= 0", "F2 SIMULADO", "interacciones_lms.duracion_min (P1), AG-01", "R-21, R-22, R-26", "Minutos LMS en P1"),
    ("sim_lms_dias_activos_p1", "int16", ">= 0", "F2 SIMULADO", "interacciones_lms.fecha (P1), AG-01", "R-26", "Días distintos con actividad en P1"),
    ("sim_lms_tareas_revisadas_p1", "int16", ">= 0", "F2 SIMULADO", "interacciones_lms.revisada (P1), AG-01", "R-26", "Tareas revisadas en P1"),
    ("sim_lms_eventos_total", "int32", ">= 0", "F2 SIMULADO", "interacciones_lms (P1–P3), AG-01", "R-26", "Eventos LMS del curso (descriptivo; no temprano)"),
    ("sim_lms_minutos_total", "int32", ">= 0", "F2 SIMULADO", "interacciones_lms (P1–P3), AG-01", "R-26", "Minutos LMS del curso (descriptivo; no temprano)"),
    ("sim_n_tutorias", "int16", ">= 0", "F3 SIMULADO", "seguimiento_riesgo.tutorias, AG-02", "R-23, R-26", "Tutorías recibidas (P2–P3; posteriores al corte)"),
    ("sim_n_tutorias_completadas", "int16", ">= 0", "F3 SIMULADO", "tutorias.resultado = completada, AG-02", "R-23, R-26", "Tutorías completadas"),
    ("sim_n_alertas", "int16", ">= 0", "F3 SIMULADO", "alertas_riesgo, AG-03", "R-24, R-26", "Alertas registradas (P2)"),
    ("sim_nivel_alerta_max", "int8", "0–3", "F3 SIMULADO", "alertas_riesgo.nivel, AG-03", "R-26", "0 sin alerta, 1 bajo, 2 medio, 3 alto"),
    ("sim_alerta_cerrada", "bool", "V/F", "F3 SIMULADO", "alertas_riesgo.estado, AG-03", "R-24, R-26", "Alguna alerta llegó a 'cerrada'"),
    ("particion", "category", "train, test", "Derivada", "StratifiedGroupKFold(5, seed 42), 1.er pliegue", "L-04", "Partición fijada antes del análisis, agrupada por estudiante"),
    ("archivo_origen", "category", "student-mat.csv, student-por.csv", "Linaje", "matricula.archivo_origen", "—", "Archivo UCI de origen"),
    ("fila_origen", "int32", ">= 2", "Linaje", "matricula.fila_origen", "—", "Línea del CSV de origen"),
]

# Correspondencia con el dataset esperado del avance
MAPEO_AVANCE = [
    ("estudiante_id", "estudiante_id", "Se mantiene; ahora con regla de identidad documentada (R-12)."),
    ("periodo (\"2026-1\")", "columnas nota_p1/p2/p3 + corte", "Etiqueta errónea; los períodos son P1–P3 de 2005–06 (config.PERIODOS)."),
    ("escuela, zona, tiempo_estudio", "escuela, zona, tiempo_estudio", "Sin cambios."),
    ("promedio_general", "nota_p1 (y nota_p2 en el escenario B)", "Eliminado: incluía G3 (fuga de información)."),
    ("faltas_totales", "faltas", "Por matrícula; anual (no se usa en el modelo temprano, L-02)."),
    ("reprobaciones_totales", "reprobaciones_previas", "Por matrícula (failures depende de la asignatura)."),
    ("n_interacciones_lms", "sim_lms_eventos_p1 / sim_lms_eventos_total", "Prefijo sim_: variable simulada."),
    ("minutos_lms", "sim_lms_minutos_p1 / sim_lms_minutos_total", "Prefijo sim_: variable simulada."),
    ("n_tutorias", "sim_n_tutorias", "Posterior al corte P1: solo descriptiva."),
    ("en_riesgo", "en_riesgo", "Definida: G3 < 10 (incluye no evaluados)."),
]


def leer_sql() -> pd.DataFrame:
    """Lee la vista de integración de PostgreSQL."""
    eng = create_engine(pg_url())
    with eng.connect() as con:
        return pd.read_sql(text("SELECT * FROM academico.v_matricula_ancha"), con)


def leer_mongo() -> dict[str, pd.DataFrame]:
    """Ejecuta los pipelines de extracción de variables AG-01/02/03."""
    db = MongoClient(MONGO_URI)[MONGO_DB]
    pipes = json.loads((MONGO_DIR / "pipelines.json").read_text(encoding="utf-8"))
    out = {}
    for pid in ("AG-01_lms_por_matricula_periodo", "AG-02_tutorias_por_matricula",
                "AG-03_alertas_por_matricula"):
        spec = pipes[pid]
        out[pid[:5]] = pd.DataFrame(list(db[spec["coleccion"]].aggregate(spec["pipeline"])))
    return out


def construir(sql: pd.DataFrame, mg: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Integra SQL y Mongo en una fila por matrícula y deriva variables."""
    k = ["estudiante_id", "asignatura"]
    df = sql.copy()
    lms = mg["AG-01"]
    p1 = lms[lms.periodo == "P1"].rename(columns={
        "n_eventos": "sim_lms_eventos_p1", "minutos": "sim_lms_minutos_p1",
        "dias_activos": "sim_lms_dias_activos_p1", "n_tareas_revisadas": "sim_lms_tareas_revisadas_p1"})
    tot = lms.groupby(k, as_index=False).agg(sim_lms_eventos_total=("n_eventos", "sum"),
                                              sim_lms_minutos_total=("minutos", "sum"))
    tut = mg["AG-02"].rename(columns={"n_tutorias": "sim_n_tutorias",
                                      "n_tutorias_completadas": "sim_n_tutorias_completadas"})
    ale = mg["AG-03"].rename(columns={"n_alertas": "sim_n_alertas",
                                      "nivel_alerta_max": "sim_nivel_alerta_max",
                                      "alerta_cerrada": "sim_alerta_cerrada"})
    n0 = len(df)
    df = (df.merge(p1[k + ["sim_lms_eventos_p1", "sim_lms_minutos_p1", "sim_lms_dias_activos_p1",
                           "sim_lms_tareas_revisadas_p1"]], on=k, how="left")
            .merge(tot, on=k, how="left")
            .merge(tut[k + ["sim_n_tutorias", "sim_n_tutorias_completadas"]], on=k, how="left")
            .merge(ale[k + ["sim_n_alertas", "sim_nivel_alerta_max", "sim_alerta_cerrada"]], on=k, how="left"))
    assert len(df) == n0, "La integración duplicó filas"
    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    n_sin_lms_p1 = int(df.sim_lms_eventos_p1.isna().sum())
    n_sin_tut = int(df.sim_n_tutorias.isna().sum())
    n_sin_al = int(df.sim_n_alertas.isna().sum())
    df[sim_cols] = df[sim_cols].fillna(0)
    regla("R-26", "Integración", "Matrículas sin registros simulados tras el LEFT JOIN "
          f"(sin LMS en P1: {n_sin_lms_p1}; sin tutorías: {n_sin_tut}; sin alertas: {n_sin_al})",
          "Imputar 0 (ausencia de registro = ausencia de actividad)",
          n_sin_lms_p1 + n_sin_tut + n_sin_al, f"{IMPL}::construir", n0, len(df))

    # Variables derivadas
    df["reprobaciones_3mas"] = df.reprobaciones_previas == 3
    df["faltas_atipicas"] = False
    for a, g in df.groupby("asignatura"):
        q1, q3 = g.faltas.quantile([0.25, 0.75])
        df.loc[g.index, "faltas_atipicas"] = g.faltas > q3 + 3 * (q3 - q1)
    df["en_riesgo"] = (df.nota_p3 < UMBRAL_APROBACION).astype("int8")
    media = df[~df.no_evaluado_p3].groupby("asignatura").nota_p3.mean()
    df["rendimiento_sobre_promedio"] = pd.array(
        [pd.NA if ne else bool(n > media[a]) for n, ne, a in
         zip(df.nota_p3, df.no_evaluado_p3, df.asignatura)], dtype="boolean")
    regla("DEF-01", "Integración", f"Variable objetivo en_riesgo = 1 si G3 < {UMBRAL_APROBACION} "
          f"(prevalencia {df.en_riesgo.mean():.3f})", "Derivar", int(df.en_riesgo.sum()),
          f"{IMPL}::construir")

    # L-04: partición train/test agrupada por estudiante y estratificada, fijada aquí
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(df, df.en_riesgo, groups=df.estudiante_id))
    df["particion"] = "train"
    df.loc[df.index[test_idx], "particion"] = "test"
    fuga = set(df[df.particion == "train"].estudiante_id) & set(df[df.particion == "test"].estudiante_id)
    assert not fuga, "Un estudiante aparece en train y test"
    regla("L-04", "Integración", "Partición train/test (80/20) agrupada por estudiante y "
          f"estratificada por en_riesgo; test = {len(test_idx)} matrículas",
          "Fijar antes de explorar", int(len(test_idx)), f"{IMPL}::construir")

    # Tipos explícitos y orden de columnas del diccionario
    tipos = {n: t for n, t, *_ in DICCIONARIO}
    df = df[[n for n, *_ in DICCIONARIO]].copy()
    for c, t in tipos.items():
        df[c] = df[c].astype(t)
    return df.sort_values(["estudiante_id", "asignatura"]).reset_index(drop=True)


def validar(df: pd.DataFrame) -> None:
    """Validaciones de salida (conteos, unicidad, dominios, ausencia de faltantes)."""
    assert len(df) == 1044
    assert not df.duplicated(["estudiante_id", "asignatura"]).any()
    assert df.estudiante_id.nunique() == 674
    obligatorias = [c for c in df.columns if c != "rendimiento_sobre_promedio"]
    assert df[obligatorias].notna().all().all()
    assert df.nota_p1.between(0, 20).all() and df.nota_p3.between(0, 20).all()
    assert set(df.en_riesgo.unique()) <= {0, 1}


def run() -> pd.DataFrame:
    """Ejecuta la integración y persiste dataset, diccionario y linaje."""
    ensure_dirs()
    df = construir(leer_sql(), leer_mongo())
    validar(df)
    pq = DATA_PROCESSED / "dataset_analitico.parquet"
    df.to_parquet(pq, index=False)
    csv = DATA_PROCESSED / "dataset_analitico.csv"
    df.to_csv(csv, index=False)
    dic = pd.DataFrame(DICCIONARIO, columns=["variable", "tipo", "dominio", "origen", "fuente_campo",
                                             "regla", "descripcion"])
    dic.to_csv(DATA_PROCESSED / "diccionario_datos.csv", index=False)
    dic[["variable", "origen", "fuente_campo", "regla"]].to_csv(DATA_PROCESSED / "linaje.csv", index=False)
    pd.DataFrame(MAPEO_AVANCE, columns=["variable_avance", "variable_final", "decision"]).to_csv(
        DATA_PROCESSED / "mapeo_avance_final.csv", index=False)
    for p in (pq, csv):
        artefacto(p, len(df))
    artefacto(DATA_PROCESSED / "diccionario_datos.csv", len(dic))
    etapa("integrate", filas=len(df), columnas=df.shape[1], estudiantes=int(df.estudiante_id.nunique()),
          en_riesgo=int(df.en_riesgo.sum()), prevalencia=round(float(df.en_riesgo.mean()), 4),
          train=int((df.particion == "train").sum()), test=int((df.particion == "test").sum()),
          variables_simuladas=len([c for c in df.columns if c.startswith("sim_")]))
    return df


if __name__ == "__main__":
    run()
