"""TRANSFORM (F1) — calidad, resolución de identidad y normalización 3FN del registro UCI.

Entradas : data/staging/uci_raw.parquet (salida de extract.py)
Salidas  : data/staging/{escuela,asignatura,periodo,estudiante,matricula,calificacion}.parquet
           data/staging/identidad.parquet (fila UCI -> estudiante_id, con evidencia)

Cada regla de calidad (R-02 … R-14) queda registrada en logs/reglas_calidad.csv con
su conteo de registros afectados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import (ASIGNATURAS, ATRIBUTOS_DESAMBIGUACION, CLAVE_IDENTIDAD, DATA_STAGING,
                    ESCUELAS, PERIODOS, UMBRAL_DESAMBIGUACION)
from tracking import artefacto, etapa, regla

IMPL = "src/transform.py"

# Dominios categóricos válidos (documentación UCI)
DOMINIOS_CAT = {
    "school": {"GP", "MS"}, "sex": {"F", "M"}, "address": {"U", "R"},
    "famsize": {"LE3", "GT3"}, "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    **{c: {"yes", "no"} for c in ["schoolsup", "famsup", "paid", "activities", "nursery",
                                   "higher", "internet", "romantic"]},
}
# Rangos numéricos válidos (documentación UCI)
DOMINIOS_NUM = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
    "studytime": (1, 4), "failures": (0, 3), "famrel": (1, 5), "freetime": (1, 5),
    "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 93), "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
}
BINARIAS = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
            "romantic"]

# Renombrado a español (snake_case) de atributos personales -> tabla estudiante
MAPA_ESTUDIANTE = {
    "sex": "sexo", "age": "edad", "address": "zona", "famsize": "tamano_familia",
    "Pstatus": "estado_padres", "Medu": "educ_madre", "Fedu": "educ_padre",
    "Mjob": "trabajo_madre", "Fjob": "trabajo_padre", "reason": "motivo_eleccion",
    "guardian": "tutor_legal", "traveltime": "tiempo_viaje", "studytime": "tiempo_estudio",
    "schoolsup": "apoyo_escolar", "famsup": "apoyo_familiar", "activities": "actividades",
    "nursery": "guarderia", "higher": "desea_superior", "internet": "internet",
    "romantic": "relacion_romantica", "famrel": "relacion_familiar", "freetime": "tiempo_libre",
    "goout": "salidas", "Dalc": "alcohol_semana", "Walc": "alcohol_finde", "health": "salud",
}
ETIQUETAS = {
    "zona": {"U": "Urbana", "R": "Rural"},
    "tamano_familia": {"LE3": "<=3", "GT3": ">3"},
    "estado_padres": {"T": "Juntos", "A": "Separados"},
    "trabajo_madre": {"teacher": "docente", "health": "salud", "services": "servicios",
                      "at_home": "hogar", "other": "otro"},
    "trabajo_padre": {"teacher": "docente", "health": "salud", "services": "servicios",
                      "at_home": "hogar", "other": "otro"},
    "motivo_eleccion": {"home": "cercania", "reputation": "reputacion", "course": "oferta",
                        "other": "otro"},
    "tutor_legal": {"mother": "madre", "father": "padre", "other": "otro"},
}


# --------------------------------------------------------------------------- #
# Reglas de diagnóstico (no modifican datos)
# --------------------------------------------------------------------------- #
def diagnostico(raw: pd.DataFrame) -> None:
    """R-02 a R-05: faltantes, duplicados exactos y dominios categóricos/numéricos."""
    cols = list(DOMINIOS_CAT) + list(DOMINIOS_NUM)
    n_na = int(raw[cols].isna().sum().sum())
    regla("R-02", "F1 UCI", "Valores faltantes en las 33 columnas", "Diagnosticar (imputar si >0)",
          n_na, f"{IMPL}::diagnostico", len(raw), len(raw))

    n_dup = int(sum(g[cols].duplicated().sum() for _, g in raw.groupby("asignatura")))
    regla("R-03", "F1 UCI", "Filas duplicadas exactas (33 columnas) dentro de cada archivo",
          "Eliminar duplicados", n_dup, f"{IMPL}::diagnostico", len(raw), len(raw) - n_dup)

    viol_cat = sum(int((~raw[c].isin(dom)).sum()) for c, dom in DOMINIOS_CAT.items())
    regla("R-04", "F1 UCI", "Valores categóricos fuera del catálogo UCI",
          "Validar/abortar", viol_cat, f"{IMPL}::diagnostico", len(raw), len(raw))

    viol_num = sum(int((~raw[c].between(lo, hi)).sum()) for c, (lo, hi) in DOMINIOS_NUM.items())
    regla("R-05", "F1 UCI", "Valores numéricos fuera de rango documentado",
          "Validar/abortar", viol_num, f"{IMPL}::diagnostico", len(raw), len(raw))
    if viol_cat or viol_num:
        raise ValueError("Violaciones de dominio en F1: revisar antes de continuar")


# --------------------------------------------------------------------------- #
# Resolución de identidad (R-11, R-12, R-13)
# --------------------------------------------------------------------------- #
def _acuerdo(a: pd.Series, b: pd.Series) -> int:
    """Número de atributos personales coincidentes entre dos filas."""
    return int(sum(a[c] == b[c] for c in ATRIBUTOS_DESAMBIGUACION))


def resolver_identidad(raw: pd.DataFrame) -> pd.DataFrame:
    """Asigna `estudiante_id` a cada fila UCI.

    Regla (dos etapas):
      1. Candidatos = filas MAT y POR con la misma clave de 13 atributos (Cortez, 2008).
      2. Dentro de cada clave se resuelve una asignación 1:1 que maximiza el número de
         coincidencias en 14 atributos personales (algoritmo húngaro). Un par se acepta
         solo si alcanza >= UMBRAL_DESAMBIGUACION coincidencias y el máximo es único.
    Las filas que comparten clave dentro del mismo archivo NO se fusionan (son
    estudiantes distintos: tienen notas distintas).
    IDs: filas POR en orden 1..649; filas MAT no emparejadas a continuación.
    """
    df = raw.copy()
    df["clave"] = df[CLAVE_IDENTIDAD].astype(str).agg("|".join, axis=1)
    mat = df[df.asignatura == "MAT"]
    por = df[df.asignatura == "POR"]

    # R-11: claves no únicas dentro de un mismo archivo
    dup_intra = sum(int(g.duplicated("clave", keep=False).sum()) for g in (mat, por))
    grupos_intra = sum(int((g.clave.value_counts() > 1).sum()) for g in (mat, por))
    perdidas_si_fusion = sum(len(g) - g.clave.nunique() for g in (mat, por))
    regla("R-11", "F1 UCI",
          f"Clave de 13 atributos repetida dentro del mismo archivo ({grupos_intra} grupos); "
          f"fusionarlas eliminaría {perdidas_si_fusion} matrículas reales",
          "No fusionar: cada fila es una matrícula distinta", dup_intra,
          f"{IMPL}::resolver_identidad", len(df), len(df))

    pares, evidencia = [], []
    claves_comunes = sorted(set(mat.clave) & set(por.clave))
    n_etapa1 = n_etapa2 = n_rechazados = 0
    for k in claves_comunes:
        A, B = mat[mat.clave == k], por[por.clave == k]
        M = np.array([[_acuerdo(a, b) for _, b in B.iterrows()] for _, a in A.iterrows()])
        filas, cols = linear_sum_assignment(-M)
        for i, j in zip(filas, cols):
            score = int(M[i, j])
            unico = (M[i, :] == score).sum() == 1 and (M[:, j] == score).sum() == 1
            ok = score >= UMBRAL_DESAMBIGUACION and unico
            etapa_n = 1 if (len(A) == 1 and len(B) == 1) else 2
            evidencia.append({"clave": k, "idx_mat": A.index[i], "idx_por": B.index[j],
                              "coincidencias": score, "etapa": etapa_n, "aceptado": ok,
                              "candidatos_mat": len(A), "candidatos_por": len(B)})
            if ok:
                pares.append((A.index[i], B.index[j]))
                n_etapa1 += etapa_n == 1
                n_etapa2 += etapa_n == 2
            else:
                n_rechazados += 1
    filas_merge_ingenuo = int(sum(
        (mat.clave == k).sum() * (por.clave == k).sum() for k in claves_comunes))

    # Asignación determinista de IDs
    ids = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ids.loc[por.index] = np.arange(1, len(por) + 1)
    for im, ip in pares:
        ids.loc[im] = ids.loc[ip]
    sin_par = [i for i in mat.index if pd.isna(ids.loc[i])]
    ids.loc[sin_par] = np.arange(len(por) + 1, len(por) + 1 + len(sin_par))
    df["estudiante_id"] = ids.astype(int)

    regla("R-12", "F1 UCI",
          f"Emparejamiento MAT-POR: {len(claves_comunes)} claves comunes; merge ingenuo = "
          f"{filas_merge_ingenuo} filas; pares aceptados = {len(pares)} "
          f"(etapa 1: {n_etapa1}, etapa 2: {n_etapa2}); rechazados = {n_rechazados}",
          "Asignar el mismo estudiante_id al par", len(pares), f"{IMPL}::resolver_identidad",
          len(df), int(df.estudiante_id.nunique()))

    # R-13: consistencia de atributos personales dentro de cada par aceptado
    conflictos = 0
    for im, ip in pares:
        conflictos += len(ATRIBUTOS_DESAMBIGUACION) - _acuerdo(df.loc[im], df.loc[ip])
    regla("R-13", "F1 UCI", "Conflictos en atributos personales entre filas del mismo estudiante",
          "Prevalece POR (archivo más completo) y se registra", conflictos,
          f"{IMPL}::resolver_identidad")

    # R-12b: posibles falsos negativos = filas sin pareja que coinciden en 26 de 27
    # atributos (13 de clave + 14 personales) con una fila sin pareja del otro archivo.
    cnt = df.estudiante_id.value_counts()
    solo = df.estudiante_id.map(cnt) == 1
    cols = CLAVE_IDENTIDAD + ATRIBUTOS_DESAMBIGUACION
    A = df[(df.asignatura == "MAT") & solo]
    B = df[(df.asignatura == "POR") & solo]
    S = (A[cols].astype(str).values[:, None, :] == B[cols].astype(str).values[None, :, :]).sum(2)
    casi = []
    for i, j in zip(*np.where(S == len(cols) - 1)):
        dif = [c for c in cols if str(A.iloc[i][c]) != str(B.iloc[j][c])]
        casi.append({"fila_mat": int(A.iloc[i].fila_origen), "fila_por": int(B.iloc[j].fila_origen),
                     "atributo_distinto": dif[0], "valor_mat": str(A.iloc[i][dif[0]]),
                     "valor_por": str(B.iloc[j][dif[0]])})
    pd.DataFrame(casi).to_csv(DATA_STAGING / "identidad_posibles_falsos_negativos.csv", index=False)
    regla("R-12b", "F1 UCI",
          "Posibles falsos negativos: filas sin pareja que coinciden en 26/27 atributos con "
          "una fila sin pareja del otro archivo", "No fusionar (regla estricta); declarar y "
          "acotar el N de estudiantes", len(casi), f"{IMPL}::resolver_identidad")

    ev = pd.DataFrame(evidencia)
    ev.to_parquet(DATA_STAGING / "identidad_evidencia.parquet", index=False)
    etapa("identidad", filas_uci=len(df), claves_comunes=len(claves_comunes),
          claves_distintas_mat=int(mat.clave.nunique()), claves_distintas_por=int(por.clave.nunique()),
          claves_distintas_union=int(df.clave.nunique()), grupos_clave_repetida=grupos_intra,
          filas_perdidas_si_fusion=perdidas_si_fusion,
          filas_merge_ingenuo=filas_merge_ingenuo, pares_aceptados=len(pares),
          pares_etapa1=n_etapa1, pares_etapa2=n_etapa2, candidatos_rechazados=n_rechazados,
          estudiantes=int(df.estudiante_id.nunique()), solo_mat=len(sin_par),
          solo_por=int(len(por) - len(pares)))
    return df


# --------------------------------------------------------------------------- #
# Transformaciones y normalización (R-06 … R-10, R-14)
# --------------------------------------------------------------------------- #
def normalizar(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construye las seis tablas 3FN a partir del staging con identidad resuelta."""
    # Catálogos
    escuela = pd.DataFrame({"escuela_id": [1, 2], "codigo": list(ESCUELAS),
                            "nombre": list(ESCUELAS.values())})
    asignatura = pd.DataFrame({"asignatura_id": [1, 2], "codigo": list(ASIGNATURAS),
                               "nombre": list(ASIGNATURAS.values())})
    periodo = pd.DataFrame([{"periodo_id": v["orden"], "codigo": k, "nombre": v["nombre"],
                             "orden": v["orden"], "nota_uci": v["nota"],
                             "fecha_inicio": v["inicio"], "fecha_fin": v["fin"]}
                            for k, v in PERIODOS.items()])
    esc_id = dict(zip(escuela.codigo, escuela.escuela_id))
    asig_id = dict(zip(asignatura.codigo, asignatura.asignatura_id))

    # R-06: estandarización de tipos y etiquetas
    d = df.copy()
    for c in BINARIAS:
        d[c] = d[c].map({"yes": True, "no": False}).astype(bool)
    regla("R-06", "F1 UCI",
          f"Estandarización: {len(BINARIAS)} binarias yes/no -> booleano; códigos -> etiquetas "
          "en español; nombres snake_case", "Transformar", len(d), f"{IMPL}::normalizar",
          len(d), len(d))

    # Tabla estudiante: una fila por estudiante_id (prevalece POR, R-13)
    d["_orden_fuente"] = (d.asignatura == "MAT").astype(int)  # POR primero
    est = (d.sort_values(["estudiante_id", "_orden_fuente"])
             .drop_duplicates("estudiante_id", keep="first"))
    estudiante = est[["estudiante_id", "school"] + list(MAPA_ESTUDIANTE)].rename(
        columns={**MAPA_ESTUDIANTE, "school": "escuela_id"})
    estudiante["escuela_id"] = estudiante.escuela_id.map(esc_id).astype("int16")
    for col, mapa in ETIQUETAS.items():
        estudiante[col] = estudiante[col].map(mapa)
    estudiante = estudiante.sort_values("estudiante_id").reset_index(drop=True)

    # R-07: failures truncado (3 representa "3 o más")
    n_trunc = int((d.failures == 3).sum())
    regla("R-07", "F1 UCI", "Reprobaciones previas truncadas: valor 3 codifica '3 o más'",
          "Conservar como ordinal 0-3 (3 = 3+); derivar reprobaciones_3mas en el dataset", n_trunc,
          f"{IMPL}::normalizar")

    # R-08: nota 0 = no evaluado (abandono/no presentado), no nota real
    notas = {"G1": "P1", "G2": "P2", "G3": "P3"}
    ceros = {g: int((d[g] == 0).sum()) for g in notas}
    regla("R-08", "F1 UCI",
          f"Nota 0 interpretada como 'no evaluado' (G1={ceros['G1']}, G2={ceros['G2']}, "
          f"G3={ceros['G3']})", "Marcar no_evaluado; excluir de medias; G3=0 cuenta como "
          "reprobación en en_riesgo", sum(ceros.values()), f"{IMPL}::normalizar")

    # R-09: faltas = 0 en matrículas no evaluadas al final -> faltas no confiables
    no_conf = (d.G3 == 0) & (d.absences == 0)
    regla("R-09", "F1 UCI",
          f"Faltas = 0 en matrículas con G3 no evaluado ({int((d.G3 == 0).sum())} con G3=0, "
          f"{int(no_conf.sum())} con 0 faltas): registro de asistencia no confiable",
          "Marcar faltas_confiable = false", int(no_conf.sum()), f"{IMPL}::normalizar")

    # R-10: atípicos en faltas (valla extrema de Tukey por asignatura), sin eliminar
    atip = pd.Series(False, index=d.index)
    for a, g in d.groupby("asignatura"):
        q1, q3 = g.absences.quantile([0.25, 0.75])
        atip.loc[g.index] = g.absences > q3 + 3 * (q3 - q1)
    d["faltas_atipicas"] = atip
    regla("R-10", "F1 UCI", "Faltas extremas (> Q3 + 3·RIC por asignatura)",
          "Conservar (valores plausibles) y marcar; log1p en modelos", int(atip.sum()),
          f"{IMPL}::normalizar")

    matricula = pd.DataFrame({
        "estudiante_id": d.estudiante_id.astype("int32"),
        "asignatura_id": d.asignatura.map(asig_id).astype("int16"),
        "reprobaciones_previas": d.failures.astype("int16"),
        "clases_pagadas": d.paid.astype(bool),
        "faltas": d.absences.astype("int16"),
        "faltas_confiable": ~no_conf,
        "archivo_origen": d.archivo_origen,
        "fila_origen": d.fila_origen.astype("int32"),
    }).sort_values(["estudiante_id", "asignatura_id"]).reset_index(drop=True)

    # R-14: 1FN — G1/G2/G3 pasan a filas de calificación
    cal = []
    for g, p in notas.items():
        cal.append(pd.DataFrame({
            "estudiante_id": d.estudiante_id.astype("int32"),
            "asignatura_id": d.asignatura.map(asig_id).astype("int16"),
            "periodo_id": np.int16(PERIODOS[p]["orden"]),
            "nota": d[g].astype("int16"),
            "no_evaluado": d[g] == 0,
        }))
    calificacion = (pd.concat(cal, ignore_index=True)
                      .sort_values(["estudiante_id", "asignatura_id", "periodo_id"])
                      .reset_index(drop=True))
    regla("R-14", "F1 UCI", "Despivotado 1FN: G1, G2, G3 -> filas (matrícula × período)",
          "Transformar", len(calificacion), f"{IMPL}::normalizar", len(d), len(calificacion))

    return {"escuela": escuela, "asignatura": asignatura, "periodo": periodo,
            "estudiante": estudiante, "matricula": matricula, "calificacion": calificacion}


def run() -> dict[str, pd.DataFrame]:
    """Ejecuta la etapa Transform de F1 y persiste las tablas en staging."""
    raw = pd.read_parquet(DATA_STAGING / "uci_raw.parquet")
    diagnostico(raw)
    ident = resolver_identidad(raw)
    ident[["asignatura", "archivo_origen", "fila_origen", "clave", "estudiante_id"]].to_parquet(
        DATA_STAGING / "identidad.parquet", index=False)
    tablas = normalizar(ident)
    for nombre, t in tablas.items():
        out = DATA_STAGING / f"{nombre}.parquet"
        t.to_parquet(out, index=False)
        artefacto(out, len(t))
    etapa("transform_uci", **{f"filas_{k}": len(v) for k, v in tablas.items()})
    return tablas


if __name__ == "__main__":
    run()
