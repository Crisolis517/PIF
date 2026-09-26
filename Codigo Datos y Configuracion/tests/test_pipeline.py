"""Pruebas del pipeline: conteos, dominios, integridad, fuga de información y reproducibilidad.

Ejecutar después de `python run_pipeline.py`:  pytest -q
Las pruebas que requieren PostgreSQL/MongoDB se omiten si los servicios no responden.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from analysis import ESCENARIOS  # noqa: E402
from transform_synthetic import categorizar, normalizar_texto, periodo_de  # noqa: E402

DS = ROOT / "data" / "processed" / "dataset_analitico.parquet"


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(DS)


# --------------------------- conteos e identidad --------------------------- #
def test_conteos_tablas_3fn():
    st = config.DATA_STAGING
    assert len(pd.read_parquet(st / "estudiante.parquet")) == 674
    assert len(pd.read_parquet(st / "matricula.parquet")) == 1044 == 395 + 649
    assert len(pd.read_parquet(st / "calificacion.parquet")) == 3132 == 1044 * 3


def test_regla_identidad():
    ident = pd.read_parquet(config.DATA_STAGING / "identidad.parquet")
    por_est = ident.groupby("estudiante_id").asignatura.agg(list)
    assert (por_est.map(len) == 2).sum() == 370                    # estudiantes en ambas
    assert por_est.map(lambda a: sorted(a) in (["MAT"], ["POR"], ["MAT", "POR"])).all()
    ev = pd.read_parquet(config.DATA_STAGING / "identidad_evidencia.parquet")
    assert (ev[ev.aceptado].coincidencias >= config.UMBRAL_DESAMBIGUACION).all()


def test_no_se_fusionan_filas_del_mismo_archivo():
    ident = pd.read_parquet(config.DATA_STAGING / "identidad.parquet")
    assert not ident.duplicated(["estudiante_id", "asignatura"]).any()


# ------------------------------ dataset final ------------------------------ #
def test_dataset_unidad_y_faltantes(df):
    assert len(df) == 1044 and df.estudiante_id.nunique() == 674
    assert not df.duplicated(["estudiante_id", "asignatura"]).any()
    assert df.drop(columns=["rendimiento_sobre_promedio"]).notna().all().all()


def test_dataset_dominios(df):
    for c in ("nota_p1", "nota_p2", "nota_p3"):
        assert df[c].between(0, 20).all()
    assert df.reprobaciones_previas.between(0, 3).all()
    assert set(df.asignatura.unique()) == {"MAT", "POR"}
    assert set(df.escuela.unique()) == {"GP", "MS"}
    assert (df.en_riesgo == (df.nota_p3 < 10).astype(int)).all()
    assert (df.no_evaluado_p3 == (df.nota_p3 == 0)).all()
    assert (df.filter(like="sim_") >= 0).all().all()


def test_variables_simuladas_marcadas():
    dic = pd.read_csv(config.DATA_PROCESSED / "diccionario_datos.csv")
    sim = dic[dic.variable.str.startswith("sim_")]
    assert sim.origen.str.contains("SIMULADO").all() and len(sim) == 11


# --------------------------- fuga de información --------------------------- #
def test_sin_fuga_de_g3():
    for esc in ESCENARIOS.values():
        feats = esc["num"] + esc["bin"]
        assert "nota_p3" not in feats and "no_evaluado_p3" not in feats          # L-01
        assert not any(f.startswith("sim_n_") or "total" in f for f in feats)    # L-03


def test_faltas_anuales_fuera_del_modelo_principal():
    assert "faltas" not in ESCENARIOS["A_corte_P1"]["num"]                        # L-02


def test_particion_agrupada_por_estudiante(df):
    tr = set(df[df.particion == "train"].estudiante_id)
    te = set(df[df.particion == "test"].estudiante_id)
    assert not tr & te                                                            # L-04
    assert 0.15 < len(df[df.particion == "test"]) / len(df) < 0.25


# ------------------------- calidad: verdad vs detección --------------------- #
def test_reglas_detectan_exactamente_lo_inyectado():
    verdad = json.loads((config.DATA_SYNTH_RAW / "_verdad_generador.json").read_text())["defectos_inyectados"]
    det = pd.read_csv(config.LOGS / "reglas_calidad.csv").drop_duplicates("id", keep="last").set_index("id").afectados
    assert det["R-15"] == verdad["fecha_faltante"]
    assert det["R-16"] == verdad["evento_duplicado"]
    assert det["R-17"] == verdad["estudiante_inexistente"]
    assert det["R-18"] == verdad["fecha_fuera_calendario"]
    assert det["R-20"] == verdad["tipo_no_estandar"]
    assert det["R-21"] == verdad["duracion_negativa"]
    assert det["R-22"] == verdad["duracion_excesiva"]
    assert det["R-23"] == verdad["tutoria_resultado_no_estandar"] + verdad["tutoria_fecha_fuera_calendario"]
    assert det["R-24"] == verdad["alerta_actualizado_incoherente"]


def test_generador_reproducible():
    from generate_synthetic import generar
    est = pd.read_parquet(config.DATA_STAGING / "estudiante.parquet")
    mat = pd.read_parquet(config.DATA_STAGING / "matricula.parquet")
    cal = pd.read_parquet(config.DATA_STAGING / "calificacion.parquet")
    a, b = generar(est, mat, cal, seed=42), generar(est, mat, cal, seed=42)
    assert a["eventos"] == b["eventos"] and a["alertas"] == b["alertas"] and a["verdad"] == b["verdad"]


# ------------------------------ funciones puras ----------------------------- #
def test_periodo_de():
    assert periodo_de(datetime(2005, 10, 3)) == "P1"
    assert periodo_de(datetime(2006, 2, 1)) == "P2"
    assert periodo_de(datetime(2006, 5, 20)) == "P3"
    assert periodo_de(datetime(2006, 7, 15)) is None
    assert periodo_de(datetime(2005, 12, 25)) is None  # vacaciones


def test_texto():
    t = normalizar_texto("INASISTENCIAS  reiteradas; se contacta al representante. Est. con faltas")
    assert t == "inasistencias reiteradas se contacta al representante estudiante con faltas"
    assert categorizar(t) == "asistencia"
    assert categorizar(normalizar_texto("Conducta disruptiva en el aula")) == "conducta"
    assert categorizar("") == "indeterminada"


# ------------------------------ bases de datos ------------------------------ #
def _pg():
    from sqlalchemy import create_engine
    try:
        eng = create_engine(config.pg_url())
        eng.connect().close()
        return eng
    except Exception:
        pytest.skip("PostgreSQL no disponible")


def _mongo():
    from pymongo import MongoClient
    try:
        cli = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=2000)
        cli.server_info()
        return cli[config.MONGO_DB]
    except Exception:
        pytest.skip("MongoDB no disponible")


def test_sql_integridad_en_vivo():
    from sqlalchemy import text
    from load import consultas_nombradas
    eng = _pg()
    with eng.connect() as con:
        for _, q in consultas_nombradas(config.SQL_DIR / "03_integridad.sql"):
            r = con.execute(text(q)).mappings().one()
            assert r["valor"] == r["esperado"], r["control"]


def test_sql_rechaza_nota_fuera_de_dominio():
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    eng = _pg()
    with pytest.raises(IntegrityError):
        with eng.begin() as con:
            con.execute(text("UPDATE academico.calificacion SET nota = 25 WHERE estudiante_id = 1 AND periodo_id = 1"))


def test_mongo_validador_rechaza_documento_incoherente():
    from pymongo.errors import WriteError
    db = _mongo()
    malo = {"alerta_id": "AL-99998", "estudiante_id": 1, "asignatura": "MAT", "periodo": "P2", "nivel": "alto",
            "estado": "abierta", "creada_en": datetime(2006, 1, 10), "actualizado_en": datetime(2006, 1, 5),
            "historial_estados": [{"estado": "abierta", "fecha": datetime(2006, 1, 10)}], "notas": [],
            "es_simulado": True, "fuente": "test"}
    with pytest.raises(WriteError):
        db.alertas_riesgo.insert_one(malo)


def test_mongo_cobertura_total():
    db = _mongo()
    assert len(db.seguimiento_riesgo.distinct("estudiante_id")) == 674
    assert db.seguimiento_riesgo.count_documents({}) == 674 * 3
