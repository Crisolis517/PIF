"""GENERATE — datos SIMULADOS de LMS (F2), tutorías y alertas (F3).

IMPORTANTE: todo lo que produce este módulo es SIMULADO. Sirve para demostrar el
pipeline híbrido SQL + NoSQL, no para extraer hallazgos sustantivos.

Proceso generador (semilla fija SEED = 42; cobertura: 100 % de las 1.044 matrículas):

F2 — Eventos LMS por matrícula m (estudiante i, asignatura a) y período t:
    a_i   ~ N(0, 0.5²)                          efecto aleatorio del estudiante
    ε_m   ~ N(0, 0.5²)                          ruido de la matrícula
    η_m   = −0.40 + 0.30·(tiempo_estudio − 2) − 0.35·reprob_previas
            + 0.40·internet + 0.25·desea_superior + a_i + ε_m
    N_mt  ~ Poisson(12 · exp(η_m) · f_t),       f = {P1: 1.0, P2: 0.9, P3: 0.8}
    tipo  ~ Categórica(recurso .35, tarea .30, foro .25, cuestionario .10)
    dur   ~ round(LogNormal(ln 15 + 0.2·η_m, 0.6)) minutos, acotada a [1, 240]
    revisada ~ Bernoulli(0.75) solo para tareas
    fecha ~ Uniforme en días lectivos (lun–vie) del período, hora 08–22
  -> El LMS NO depende de ninguna nota (G1, G2, G3): cualquier asociación con la
     reprobación proviene solo de covariables reales (tiempo de estudio, reprobaciones
     previas, internet, aspiración) y es, por construcción, condicionalmente
     independiente de G3 dadas esas covariables.

F3 — Tutorías y alertas (disparadas al cierre de P1, información ya disponible):
    Alerta:  nivel alto si G1 < 8; medio si 8 <= G1 < 10 o reprob_previas >= 2;
             bajo con prob. 0.03 en el resto. Se registra con prob. 0.90/0.80/1.00.
             Ciclo: abierta -> en_seguimiento (prob. 0.85, +3–15 días)
                    -> cerrada (prob. 0.60/0.75/0.90 según nivel, +20–70 días).
             1–3 notas de texto libre por plantilla con ruido tipográfico.
    Tutoría: prob. 0.70 si G1 < 10; 0.15 si reprob_previas >= 1; 0.04 en otro caso.
             Sesiones = 1 + Poisson(1.2), fechas en P2–P3.

Defectos de calidad inyectados deliberadamente (tasas explícitas) para ejercitar las
reglas R-15 … R-24; su número exacto se guarda en `_verdad_generador.json` y las
pruebas verifican que el ETL detecta exactamente lo inyectado.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from config import DATA_STAGING, DATA_SYNTH_RAW, PERIODOS, SEED, ensure_dirs
from tracking import artefacto, etapa

VERSION_GENERADOR = "generate_synthetic v1.0 (seed=42)"

TIPOS = ["recurso", "tarea", "foro", "cuestionario"]
P_TIPOS = [0.35, 0.30, 0.25, 0.10]
FACTOR_PERIODO = {"P1": 1.0, "P2": 0.9, "P3": 0.8}

# Tasas de defectos inyectados
TASAS = {
    "evento_duplicado": 0.02, "tipo_no_estandar": 0.05, "duracion_negativa": 0.005,
    "duracion_excesiva": 0.005, "fecha_fuera_calendario": 0.003,
    "estudiante_inexistente": 0.002, "fecha_faltante": 0.002,
    "tutoria_resultado_no_estandar": 0.05, "tutoria_fecha_fuera_calendario": 0.01,
    "alerta_actualizado_incoherente": 0.12,
}
VARIANTES_TIPO = {"recurso": ["Recurso", "RECURSO ", "resource"],
                  "tarea": ["Tarea", " tarea", "task", "TAREA"],
                  "foro": ["Foro", "FORO ", "forum"],
                  "cuestionario": ["Cuestionario", "quiz", "QUIZ"]}
VARIANTES_RESULTADO = {"completada": ["Completada", "COMPLETADA", "completed"],
                       "no_asistio": ["No asistió", "no-asistio"],
                       "reprogramada": ["Reprogramada", "rescheduled"]}

PLANTILLAS = {
    "academico": [
        "El estudiante presenta bajo rendimiento en {asig}; se recomienda refuerzo en los contenidos del período.",
        "Dificultades en la resolución de ejercicios y en la entrega de tareas de {asig}.",
        "Nota del primer período por debajo del mínimo; se acuerda un plan de recuperación.",
    ],
    "asistencia": [
        "Inasistencias reiteradas a clase; se contacta al representante para justificar las faltas.",
        "Retrasos y ausencias frecuentes en la jornada de {asig}.",
        "Registro de asistencia irregular durante el período; se solicita justificación.",
    ],
    "familiar": [
        "Situación familiar compleja reportada por el representante; se deriva a orientación.",
        "La familia informa cambios en el hogar que afectan al estudiante.",
    ],
    "conducta": [
        "Conducta disruptiva en el aula; se registra un llamado de atención.",
        "Conflictos con compañeros durante la clase de {asig}.",
    ],
    "motivacion": [
        "Baja motivación y desinterés por la asignatura; manifiesta no querer continuar estudios.",
        "El estudiante expresa desánimo y falta de metas académicas.",
    ],
}
NOMBRE_ASIG = {"MAT": "Matemáticas", "POR": "Lengua Portuguesa"}
TUTORES = {1: [f"TUT-{i:02d}" for i in range(1, 6)], 2: [f"TUT-{i:02d}" for i in range(6, 9)]}
ORIENTADORES = [f"ORI-{i:02d}" for i in range(1, 5)]


def _dias_lectivos(p: str) -> list[date]:
    ini, fin = PERIODOS[p]["inicio"], PERIODOS[p]["fin"]
    dias = [ini + timedelta(d) for d in range((fin - ini).days + 1)]
    return [d for d in dias if d.weekday() < 5]


def _instante(rng: np.random.Generator, dias: list[date]) -> datetime:
    d = dias[rng.integers(len(dias))]
    return datetime(d.year, d.month, d.day, int(rng.integers(8, 22)), int(rng.integers(0, 60)))


def _ruido_texto(rng: np.random.Generator, texto: str) -> str:
    """Ruido tipográfico realista para ejercitar la normalización de texto libre."""
    if rng.random() < 0.30:
        texto = (texto.replace("á", "a").replace("é", "e").replace("í", "i")
                 .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
    if rng.random() < 0.15:
        texto = texto.replace("estudiante", "est.").replace(" que ", " q ")
    if rng.random() < 0.20:
        texto = texto.replace(" ", "  ", 2)
    if rng.random() < 0.10:
        texto = texto.upper()
    return texto


def generar(est: pd.DataFrame, mat: pd.DataFrame, cal: pd.DataFrame, seed: int = SEED) -> dict:
    """Genera eventos LMS, tutorías y alertas (con defectos inyectados)."""
    rng = np.random.default_rng(seed)
    asig_cod = {1: "MAT", 2: "POR"}
    g1 = cal[cal.periodo_id == 1].set_index(["estudiante_id", "asignatura_id"]).nota
    df = mat.merge(est[["estudiante_id", "escuela_id", "tiempo_estudio", "internet",
                        "desea_superior"]], on="estudiante_id")
    df["g1"] = [int(g1.loc[(e, a)]) for e, a in zip(df.estudiante_id, df.asignatura_id)]
    df["asignatura"] = df.asignatura_id.map(asig_cod)

    # ---------------- F2: LMS ----------------
    a_i = dict(zip(est.estudiante_id, rng.normal(0, 0.5, len(est))))
    eps = rng.normal(0, 0.5, len(df))
    df["eta"] = (-0.40 + 0.30 * (df.tiempo_estudio - 2) - 0.35 * df.reprobaciones_previas
                 + 0.40 * df.internet.astype(int) + 0.25 * df.desea_superior.astype(int)
                 + df.estudiante_id.map(a_i) + eps)
    dias = {p: _dias_lectivos(p) for p in PERIODOS}
    eventos, n_ev = [], 0
    for r in df.itertuples():
        for p in PERIODOS:
            n = rng.poisson(12 * np.exp(r.eta) * FACTOR_PERIODO[p])
            for _ in range(n):
                n_ev += 1
                tipo = TIPOS[rng.choice(4, p=P_TIPOS)]
                dur = min(240, max(1, int(round(rng.lognormal(np.log(15) + 0.2 * r.eta, 0.6)))))
                ev = {"evento_id": f"EV-{n_ev:07d}", "estudiante_id": int(r.estudiante_id),
                      "asignatura": r.asignatura, "fecha": _instante(rng, dias[p]).isoformat(),
                      "tipo": tipo, "duracion_min": dur}
                if tipo == "tarea":
                    ev["revisada"] = bool(rng.random() < 0.75)
                eventos.append(ev)
    n_limpios = len(eventos)

    # Defectos inyectados en F2 (índices disjuntos para que el conteo sea exacto)
    verdad = {"eventos_generados": n_limpios}
    idx = rng.permutation(n_limpios)
    cortes, pos = {}, 0
    for k in ["tipo_no_estandar", "duracion_negativa", "duracion_excesiva",
              "fecha_fuera_calendario", "estudiante_inexistente", "fecha_faltante"]:
        n_k = int(round(TASAS[k] * n_limpios))
        cortes[k] = idx[pos:pos + n_k]
        pos += n_k
    for i in cortes["tipo_no_estandar"]:
        t = eventos[i]["tipo"]
        eventos[i]["tipo"] = VARIANTES_TIPO[t][rng.integers(len(VARIANTES_TIPO[t]))]
    for i in cortes["duracion_negativa"]:
        eventos[i]["duracion_min"] = -eventos[i]["duracion_min"]
    for i in cortes["duracion_excesiva"]:
        eventos[i]["duracion_min"] = int(rng.integers(300, 1500))
    for i in cortes["fecha_fuera_calendario"]:
        eventos[i]["fecha"] = datetime(2006, 7, int(rng.integers(1, 31)), 10, 0).isoformat()
    for i in cortes["estudiante_inexistente"]:
        eventos[i]["estudiante_id"] = int(9000 + rng.integers(0, 999))
    for i in cortes["fecha_faltante"]:
        eventos[i]["fecha"] = None
    for k, v in cortes.items():
        verdad[k] = int(len(v))
    # Duplicados: copias exactas de eventos sin otros defectos
    sanos = idx[pos:]
    dup = rng.choice(sanos, int(round(TASAS["evento_duplicado"] * n_limpios)), replace=False)
    eventos.extend(dict(eventos[i]) for i in dup)
    verdad["evento_duplicado"] = int(len(dup))
    orden = rng.permutation(len(eventos))  # los logs llegan desordenados
    eventos = [eventos[i] for i in orden]

    # ---------------- F3: tutorías ----------------
    tutorias, n_tut = [], 0
    dias_p23 = dias["P2"] + dias["P3"]
    for r in df.itertuples():
        p = 0.70 if r.g1 < 10 else (0.15 if r.reprobaciones_previas >= 1 else 0.04)
        if rng.random() >= p:
            continue
        k = 1 + rng.poisson(1.2)
        fechas = sorted(_instante(rng, dias_p23) for _ in range(k))
        for f in fechas:
            n_tut += 1
            tutorias.append({
                "tutoria_id": f"TU-{n_tut:05d}", "estudiante_id": int(r.estudiante_id),
                "asignatura": r.asignatura, "fecha": f.isoformat(),
                "tutor": TUTORES[r.escuela_id][rng.integers(len(TUTORES[r.escuela_id]))],
                "modalidad": ["presencial", "virtual"][int(rng.random() < 0.4)],
                "tema": rng.choice(["rendimiento_academico", "asistencia", "tecnicas_estudio",
                                    "orientacion_personal"], p=[0.45, 0.20, 0.20, 0.15]).item(),
                "resultado": rng.choice(["completada", "no_asistio", "reprogramada"],
                                        p=[0.70, 0.15, 0.15]).item(),
            })
    it = rng.permutation(len(tutorias))
    n_res = int(round(TASAS["tutoria_resultado_no_estandar"] * len(tutorias)))
    n_fec = int(round(TASAS["tutoria_fecha_fuera_calendario"] * len(tutorias)))
    for i in it[:n_res]:
        v = VARIANTES_RESULTADO[tutorias[i]["resultado"]]
        tutorias[i]["resultado"] = v[rng.integers(len(v))]
    for i in it[n_res:n_res + n_fec]:
        tutorias[i]["fecha"] = datetime(2006, 8, int(rng.integers(1, 31)), 11, 0).isoformat()
    verdad.update(tutorias_generadas=len(tutorias), tutoria_resultado_no_estandar=n_res,
                  tutoria_fecha_fuera_calendario=n_fec)

    # ---------------- F3: alertas ----------------
    alertas, cat_verdad, n_al = [], {}, 0
    ini_p2 = PERIODOS["P2"]["inicio"]
    for r in df.itertuples():
        if r.g1 < 8:
            nivel, p_reg = "alto", 0.90
        elif r.g1 < 10 or r.reprobaciones_previas >= 2:
            nivel, p_reg = "medio", 0.80
        elif rng.random() < 0.03:
            nivel, p_reg = "bajo", 1.0
        else:
            continue
        if rng.random() >= p_reg:
            continue
        n_al += 1
        creada = datetime.combine(ini_p2 + timedelta(int(rng.integers(0, 18))), datetime.min.time()) \
            + timedelta(hours=int(rng.integers(8, 17)))
        hist = [{"estado": "abierta", "fecha": creada}]
        if rng.random() < 0.85:
            f2 = creada + timedelta(days=int(rng.integers(3, 16)))
            hist.append({"estado": "en_seguimiento", "fecha": f2})
            p_cierre = {"alto": 0.60, "medio": 0.75, "bajo": 0.90}[nivel]
            if rng.random() < p_cierre:
                hist.append({"estado": "cerrada",
                             "fecha": f2 + timedelta(days=int(rng.integers(20, 71)))})
        # categoría de las notas: la asistencia es más probable si hay muchas faltas (dato real)
        pesos = np.array([0.40, 0.15, 0.15, 0.12, 0.18])
        if r.faltas > 10:
            pesos[1] += 0.30
        pesos /= pesos.sum()
        cats = list(PLANTILLAS)
        notas = []
        ultima = hist[-1]["fecha"]
        for _ in range(int(rng.integers(1, 4))):
            c = cats[rng.choice(len(cats), p=pesos)]
            plantilla = PLANTILLAS[c][rng.integers(len(PLANTILLAS[c]))]
            dia = creada.date() + timedelta(int(rng.integers(0, (ultima - creada).days + 1)))
            f = datetime.combine(dia, datetime.min.time()) + timedelta(hours=int(rng.integers(8, 18)))
            f = min(max(f, creada), max(ultima, creada))
            nid = f"AL-{n_al:05d}-N{len(notas) + 1}"
            notas.append({"nota_id": nid, "fecha": f, "autor": ORIENTADORES[rng.integers(4)],
                          "texto": _ruido_texto(rng, plantilla.format(asig=NOMBRE_ASIG[r.asignatura]))})
            cat_verdad[nid] = c
        notas.sort(key=lambda n: n["fecha"])
        actualizado = max([h["fecha"] for h in hist] + [n["fecha"] for n in notas])
        alertas.append({
            "alerta_id": f"AL-{n_al:05d}", "estudiante_id": int(r.estudiante_id),
            "asignatura": r.asignatura, "periodo": "P2", "nivel": nivel,
            "estado": hist[-1]["estado"], "creada_en": creada, "actualizado_en": actualizado,
            "historial_estados": hist, "notas": notas,
        })
    ia = rng.permutation(len(alertas))
    n_inc = int(round(TASAS["alerta_actualizado_incoherente"] * len(alertas)))
    for i in ia[:n_inc]:  # reproduce el defecto del avance: actualizado_en < última fecha
        alertas[i]["actualizado_en"] = alertas[i]["creada_en"] - timedelta(days=int(rng.integers(1, 10)))
    verdad.update(alertas_generadas=len(alertas), alerta_actualizado_incoherente=n_inc,
                  notas_generadas=len(cat_verdad))

    def _ser(o):
        return o.isoformat() if isinstance(o, (datetime, date)) else o
    alertas_json = json.loads(json.dumps(alertas, default=_ser))
    return {"eventos": eventos, "tutorias": tutorias, "alertas": alertas_json,
            "verdad": verdad, "categorias_notas": cat_verdad,
            "eta": df[["estudiante_id", "asignatura", "eta"]]}


def run() -> dict:
    """Genera y guarda los JSON crudos simulados en data/raw/synthetic/."""
    ensure_dirs()
    est = pd.read_parquet(DATA_STAGING / "estudiante.parquet")
    mat = pd.read_parquet(DATA_STAGING / "matricula.parquet")
    cal = pd.read_parquet(DATA_STAGING / "calificacion.parquet")
    out = generar(est, mat, cal)
    p_ev = DATA_SYNTH_RAW / "lms_eventos.jsonl"
    with open(p_ev, "w", encoding="utf-8") as f:
        for e in out["eventos"]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    p_tu = DATA_SYNTH_RAW / "tutorias.jsonl"
    with open(p_tu, "w", encoding="utf-8") as f:
        for t in out["tutorias"]:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    p_al = DATA_SYNTH_RAW / "alertas.json"
    with open(p_al, "w", encoding="utf-8") as f:
        json.dump(out["alertas"], f, ensure_ascii=False, indent=1)
    p_v = DATA_SYNTH_RAW / "_verdad_generador.json"
    with open(p_v, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION_GENERADOR, "seed": SEED, "tasas": TASAS,
                   "defectos_inyectados": out["verdad"],
                   "categoria_real_por_nota": out["categorias_notas"]}, f, ensure_ascii=False,
                  indent=1)
    out["eta"].to_parquet(DATA_STAGING / "sim_eta_latente.parquet", index=False)
    for p, n in [(p_ev, len(out["eventos"])), (p_tu, len(out["tutorias"])),
                 (p_al, len(out["alertas"])), (p_v, None)]:
        artefacto(p, n)
    etapa("generate_synthetic", **out["verdad"])
    return out


if __name__ == "__main__":
    run()
