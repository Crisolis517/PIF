"""REPORT — genera tablas LaTeX y macros numéricas desde results/ y logs/.

Garantiza que ninguna cifra del informe se escribe a mano: el texto de
informe/main.tex usa las macros de informe/generated/valores.tex y las tablas
informe/generated/tab_*.tex, ambas producidas aquí a partir de salidas ejecutadas.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_STAGING, DATA_SYNTH_RAW, GENERATED_TEX, LOGS, RESULTS, ensure_dirs

ASIG = {"MAT": "Matemáticas", "POR": "Portugués"}
ESC = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}


# ------------------------------- formato ------------------------------------ #
def tex(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("→", r"$\rightarrow$"), ("≥", r"$\geq$"),
                 ("≤", r"$\leq$"), ("·", r"$\cdot$"), ("×", r"$\times$"), ("−", "-"),
                 ("δ", r"$\delta$"), ("<=", r"$\leq$"), (">=", r"$\geq$")]:
        s = s.replace(a, b)
    return s


def n(x, d=2) -> str:
    """Número con coma decimal."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", r"\,")


def e(x) -> str:
    """Entero con separador de miles fino."""
    return f"{int(x):,}".replace(",", r"\,")


def pct(x, d=1) -> str:
    return n(100 * x, d) + r"\,\%"


def pval(p) -> str:
    if p < 0.001:
        return r"$<$0,001"
    return n(p, 3)


def ic(lo, hi, d=2, porcentaje=False) -> str:
    f = (lambda v: n(100 * v, 1)) if porcentaje else (lambda v: n(v, d))
    return f"[{f(lo)}; {f(hi)}]"


def tabular(cols: str, cabecera: list[str], filas: list[list[str]], nombre: str) -> None:
    lineas = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(cabecera) + r" \\", r"\midrule"]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED_TEX / f"tab_{nombre}.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def longtable(cols: str, cabecera: list[str], filas: list[list[str]], nombre: str, caption: str, label: str) -> None:
    cab = " & ".join(cabecera) + r" \\"
    lineas = [r"{\scriptsize", rf"\begin{{longtable}}{{{cols}}}", rf"\caption{{{caption}}}\label{{{label}}}\\",
              r"\toprule", cab, r"\midrule", r"\endfirsthead", r"\toprule", cab, r"\midrule", r"\endhead"]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas += [r"\bottomrule", r"\end{longtable}}"]
    (GENERATED_TEX / f"tab_{nombre}.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def rd(nombre: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{nombre}.csv")


# ------------------------------- tablas ------------------------------------- #
def run() -> None:
    ensure_dirs()
    M: dict[str, str] = {}
    log = json.loads((LOGS / "ultima_ejecucion.json").read_text(encoding="utf-8"))
    et = log["etapas"]

    # --- Identidad y volúmenes
    idn = et["identidad"]
    M.update(nFilasUci=e(idn["filas_uci"]), nClavesComunes=e(idn["claves_comunes"]),
             nMergeIngenuo=e(idn["filas_merge_ingenuo"]), nPares=e(idn["pares_aceptados"]),
             nParesUno=e(idn["pares_etapa1"]), nParesDos=e(idn["pares_etapa2"]),
             nEst=e(idn["estudiantes"]), nSoloMat=e(idn["solo_mat"]), nSoloPor=e(idn["solo_por"]),
             nClavesMat=e(idn["claves_distintas_mat"]), nClavesPor=e(idn["claves_distintas_por"]),
             nClavesUnion=e(idn["claves_distintas_union"]),
             nMatAvance=e(idn["claves_distintas_mat"] + idn["claves_distintas_por"]),
             nGruposRep=e(idn["grupos_clave_repetida"]), nPerdidasFusion=e(idn["filas_perdidas_si_fusion"]))
    lp = et["load_postgres"]
    M.update(nMat=e(lp["filas_matricula"]), nCal=e(lp["filas_calificacion"]),
             nRegSql=e(lp["total_registros"]), pgVersion=tex(lp["servidor"]))
    lm = et["load_mongo"]
    M.update(nDocSeg=e(lm["seguimiento_riesgo"]), nDocAle=e(lm["alertas_riesgo"]),
             nCuarentena=e(lm["cuarentena"]), nEvEmb=e(lm["eventos_lms_embebidos"]),
             nTutEmb=e(lm["tutorias_embebidas"]), nEstCub=e(lm["estudiantes_cubiertos"]),
             mongoVersion=tex(lm["servidor"]))
    gs = et["generate_synthetic"]
    M.update(nEvGen=e(gs["eventos_generados"]), nEvDup=e(gs["evento_duplicado"]),
             nTutGen=e(gs["tutorias_generadas"]), nAleGen=e(gs["alertas_generadas"]),
             nNotasGen=e(gs["notas_generadas"]), nEvCrudos=e(et["clean_lms"]["eventos_crudos"]),
             nEvLimpios=e(et["clean_lms"]["eventos_limpios"]),
             nTutLimpias=e(et["clean_tutorias"]["tutorias_limpias"]))
    ig = et["integrate"]
    M.update(nFilasDs=e(ig["filas"]), nColsDs=e(ig["columnas"]), nRiesgo=e(ig["en_riesgo"]),
             prevalencia=pct(ig["prevalencia"]), nTrain=e(ig["train"]), nTest=e(ig["test"]),
             nVarSim=e(ig["variables_simuladas"]))
    ms = et["mongo_almacenamiento"]
    M.update(mbSeg=n(ms["seguimiento_riesgo_bytes"] / 1e6, 1), kbAle=n(ms["alertas_riesgo_bytes"] / 1e3, 0))

    # --- Conteos por etapa (evidencia de ejecución)
    filas = [
        ["Extract", "CSV UCI leídos (MAT + POR)", e(et["extract_MAT"]["filas_leidas"]) + " + " + e(et["extract_POR"]["filas_leidas"]), "0", e(et["extract"]["filas_totales"])],
        ["Transform F1", "Resolución de identidad", e(idn["filas_uci"]) + " filas", "0", e(idn["estudiantes"]) + " estudiantes"],
        ["Transform F1", "Normalización 3FN", e(idn["filas_uci"]) + " matrículas", "0", e(lp["filas_calificacion"]) + " calificaciones"],
        ["Generate", "Eventos LMS simulados + duplicados", e(et["clean_lms"]["eventos_crudos"]), "--", "--"],
        ["Transform F2", "Limpieza de eventos (R-15…R-22)", e(et["clean_lms"]["eventos_crudos"]),
         e(et["clean_lms"]["eventos_crudos"] - et["clean_lms"]["eventos_limpios"]), e(et["clean_lms"]["eventos_limpios"])],
        ["Transform F3", "Limpieza de tutorías (R-23)", e(et["clean_tutorias"]["tutorias_crudas"]),
         e(et["clean_tutorias"]["tutorias_crudas"] - et["clean_tutorias"]["tutorias_limpias"]), e(et["clean_tutorias"]["tutorias_limpias"])],
        ["Transform F3", "Alertas (R-24, R-25)", e(gs["alertas_generadas"]), "0", e(et["clean_alertas"]["alertas"])],
        ["Load SQL", "Seis tablas en PostgreSQL", e(lp["total_registros"]), "0", e(lp["total_registros"])],
        ["Load NoSQL", "seguimiento\\_riesgo + alertas\\_riesgo", e(lm["seguimiento_riesgo"] + lm["alertas_riesgo"]),
         e(lm["rechazados_validador"]), e(lm["seguimiento_riesgo"] + lm["alertas_riesgo"])],
        ["Integrate", "Dataset analítico (matrículas)", e(lp["filas_matricula"]), "0", e(ig["filas"])],
    ]
    tabular("llrrr", ["Etapa", "Operación", "Entrada", "Rechazados", "Salida"], filas, "conteos_etapas")
    dur = [[tex(k), n(v["duracion_s"], 2)] for k, v in et.items() if k[0].isdigit() and "duracion_s" in v]
    tabular("lr", ["Etapa", "Duración (s)"], dur, "duraciones")
    M["tiempoTotal"] = n(sum(v["duracion_s"] for k, v in et.items() if k[0].isdigit() and "duracion_s" in v), 1)

    # --- Reglas de calidad
    reglas = pd.read_csv(LOGS / "reglas_calidad.csv")
    reglas = reglas.drop_duplicates("id", keep="last")
    orden = lambda r: (0 if r.startswith("R") else 1, r)  # noqa: E731
    reglas = reglas.iloc[sorted(range(len(reglas)), key=lambda i: orden(reglas.id.iloc[i]))]
    filas = [[tex(r.id), tex(r.fuente), tex(r.descripcion), tex(r.accion), e(r.afectados),
              r"\texttt{" + tex(r.implementacion.replace("src/", "")).replace("::", r"::\allowbreak ").replace(" / ", r" /\allowbreak ") + "}"] for r in reglas.itertuples()]
    tabular(r">{\raggedright\arraybackslash}p{1.0cm}>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{8.6cm}>{\raggedright\arraybackslash}p{4.4cm}r>{\raggedright\arraybackslash}p{4.6cm}", ["ID", "Fuente", "Regla y diagnóstico", "Acción", "Afect.", "Implementación"],
            filas, "reglas")
    for r in reglas.itertuples():
        M["regla" + r.id.replace("-", "").replace("DEF", "Def")] = e(r.afectados)

    # --- Verdad del generador vs detección
    verdad = json.loads((DATA_SYNTH_RAW / "_verdad_generador.json").read_text(encoding="utf-8"))["defectos_inyectados"]
    det = {r.id: r.afectados for r in reglas.itertuples()}
    comp = [("fecha_faltante", "R-15", "Campo obligatorio (fecha) ausente"), ("evento_duplicado", "R-16", "Evento duplicado"),
            ("estudiante_inexistente", "R-17", "Estudiante inexistente"), ("fecha_fuera_calendario", "R-18", "Fecha fuera de calendario"),
            ("tipo_no_estandar", "R-20", "Tipo no estandarizado"), ("duracion_negativa", "R-21", "Duración negativa"),
            ("duracion_excesiva", "R-22", "Duración > 240 min"),
            ("tutoria_resultado_no_estandar+tutoria_fecha_fuera_calendario", "R-23", "Tutoría: resultado/fecha"),
            ("alerta_actualizado_incoherente", "R-24", "Alerta: actualizado\\_en incoherente")]
    filas = []
    for k, rid, et_ in comp:
        iny = sum(verdad[x] for x in k.split("+"))
        filas.append([et_, rid, e(iny), e(det[rid]), r"\checkmark" if iny == det[rid] else r"$\times$"])
    tabular("llrrc", ["Defecto inyectado", "Regla", "Inyectados", "Detectados", "Coincide"], filas, "verdad_generador")

    # --- Integridad SQL
    integ = rd("sql_integridad")
    filas = [[tex(r.control), e(r.valor), e(r.esperado), r"\checkmark" if r.ok else r"$\times$"] for r in integ.itertuples()]
    tabular("lrrc", ["Control", "Valor", "Esperado", "OK"], filas, "sql_integridad")
    M["nControlesSql"] = e(len(integ))
    M["nControlesOk"] = e(integ.ok.sum())

    a1 = rd("sql_A-01")
    filas = [[ASIG[r.asignatura], r.periodo, e(r.n_evaluados), n(r.media), n(r.desv_est), n(r.mediana, 0), e(r.n_no_evaluados)]
             for r in a1.itertuples()]
    tabular("llrrrrr", ["Asignatura", "Período", "n eval.", "Media", "DE", "Mediana", "No eval."], filas, "sql_a01")
    a2 = rd("sql_A-02")
    filas = [[ESC[r.escuela], ASIG[r.asignatura], e(r.n_matriculas), e(r.n_reprobados), pct(r.tasa_reprobacion), e(r.n_no_evaluados)]
             for r in a2.itertuples()]
    tabular("llrrrr", ["Escuela", "Asignatura", "Matríc.", "Reprob.", "Tasa", "No eval."], filas, "sql_a02")

    # --- Mongo
    ag4 = rd("mongo_AG-04")
    filas = [[r.nivel, e(r.n_alertas), e(r.con_tutoria_posterior), n(r.pct_con_tutoria, 1) + r"\,\%", e(r.cerradas)] for r in ag4.itertuples()]
    tabular("lrrrr", ["Nivel", "Alertas", "Con tutoría posterior", "\\%", "Cerradas"], filas, "mongo_ag04")
    M["agCuatroAlto"] = n(ag4.set_index("nivel").loc["alto", "pct_con_tutoria"], 1) + r"\,\%"
    M["agCuatroMedio"] = n(ag4.set_index("nivel").loc["medio", "pct_con_tutoria"], 1) + r"\,\%"
    ag5 = rd("mongo_AG-05").pivot(index="nivel", columns="estado", values="n").fillna(0)
    filas = [[i] + [e(ag5.loc[i].get(c, 0)) for c in ["abierta", "en_seguimiento", "cerrada"]] for i in ["alto", "medio", "bajo"] if i in ag5.index]
    tabular("lrrr", ["Nivel", "Abierta", "En seguimiento", "Cerrada"], filas, "mongo_ag05")
    ag6 = rd("mongo_AG-06").pivot(index="categoria", columns="nivel", values="n").fillna(0)
    filas = [[tex(i)] + [e(ag6.loc[i].get(c, 0)) for c in ["alto", "medio", "bajo"]] + [e(ag6.loc[i].sum())] for i in ag6.index]
    tabular("lrrrr", ["Categoría inferida", "Alto", "Medio", "Bajo", "Total"], filas, "mongo_ag06")
    ag7 = rd("mongo_AG-07")
    M["nBusquedaTexto"] = e(ag7.n_alertas.sum())

    # --- Descriptivos
    d = rd("descriptivos")
    filas = [[ASIG.get(r.asignatura, "Total"), e(r.n_matriculas), e(r.en_riesgo), pct(r.prevalencia) + " " + ic(r.ic_lo, r.ic_hi, porcentaje=True),
              tex(r.nota_p1_mediana_ric) if isinstance(r.nota_p1_mediana_ric, str) else "--",
              tex(r.nota_p3_mediana_ric) if isinstance(r.nota_p3_mediana_ric, str) else "--",
              tex(r.faltas_mediana_ric) if isinstance(r.faltas_mediana_ric, str) else "--",
              e(r.no_evaluados_p3), e(r.con_reprob_previas)] for r in d.itertuples()]
    tabular("lrrlllrrr", ["Asignatura", "n", "En riesgo", r"Prevalencia [IC 95\,\%]", "G1 Md [RIC]", "G3 Md [RIC]",
                          "Faltas Md [RIC]", "No eval. G3", "Reprob. prev."], filas, "descriptivos")
    for r in d.itertuples():
        k = {"MAT": "Mat", "POR": "Por", "Total": "Tot"}[r.asignatura]
        M[f"prev{k}"] = pct(r.prevalencia)
        M[f"prevIc{k}"] = ic(r.ic_lo, r.ic_hi, porcentaje=True)

    # --- P1
    med = rd("p1_medias_periodo")
    filas = [[ASIG[r.asignatura], r.periodo, e(r.n), n(r.media) + " " + ic(r.ic_lo, r.ic_hi), n(r.de), n(r.media_con_ceros)] for r in med.itertuples()]
    tabular("llrlrr", ["Asignatura", "Período", "n", r"Media [IC 95\,\%]", "DE", "Media con ceros"], filas, "p1_medias")
    for r in med.itertuples():
        M[f"media{r.asignatura.title()}{r.periodo}"] = n(r.media)
        M[f"mediaCeros{r.asignatura.title()}{r.periodo}"] = n(r.media_con_ceros)
    pr = rd("p1_pruebas_evolucion")
    filas = [[ASIG[r.asignatura], tex(r.comparacion), e(r.n), n(r.dif_media) + " " + ic(r.dif_ic_lo, r.dif_ic_hi), n(r.r_rb), pval(r.p_holm)]
             for r in pr.itertuples()]
    tabular("llrlrr", ["Asignatura", "Comparación", "n", r"Dif. media [IC 95\,\% bootstrap]", r"$r_{rb}$", "$p$ (Holm)"], filas, "p1_pruebas")
    for a, g in pr.groupby("asignatura"):
        r = g.iloc[0]
        k = a.title()
        M[f"friedChi{k}"] = n(r.friedman_chi2)
        M[f"friedP{k}"] = pval(r.friedman_p)
        M[f"kendallW{k}"] = n(r.kendall_w, 3)
        M[f"nFried{k}"] = e(r.n)
        t = g[g.comparacion.str.contains("P1→P3")].iloc[0]
        M[f"difPUnoTres{k}"] = n(t.dif_media)
        M[f"difPUnoTresIc{k}"] = ic(t.dif_ic_lo, t.dif_ic_hi)
    rep = rd("p1_reprobacion_escuela_asignatura")
    chi = rd("p1_chi2_escuela").set_index("asignatura")
    filas = [[ASIG[r.asignatura], ESC[r.escuela], e(r.n), e(r.reprobados), pct(r.tasa) + " " + ic(r.ic_lo, r.ic_hi, porcentaje=True),
              "Sí" if r.prioridad else "No"] for r in rep.itertuples()]
    tabular("llrrlc", ["Asignatura", "Escuela", "n", "Reprob.", r"Tasa [IC 95\,\% Wilson]", "Prioritaria"], filas, "p1_reprobacion")
    for r in rep.itertuples():
        M[f"tasa{r.asignatura.title()}{r.escuela.title()}"] = pct(r.tasa)
        M[f"tasaIc{r.asignatura.title()}{r.escuela.title()}"] = ic(r.ic_lo, r.ic_hi, porcentaje=True)
    for a, r in chi.iterrows():
        k = a.title()
        M[f"chiEsc{k}"] = n(r.chi2)
        M[f"chiEscP{k}"] = pval(r.p_holm)
        M[f"vEsc{k}"] = n(r.v_cramer, 3)
        M[f"difEsc{k}"] = n(100 * r.dif_tasas_MS_GP, 1)
    per = rd("p1_persistencia")
    filas = [[ASIG[r.asignatura], "Reprueba P1" if r.estado_p1 == "reprueba_P1" else "Aprueba P1", e(r.n), e(r.reprueban_p3),
              pct(r.tasa) + " " + ic(r.ic_lo, r.ic_hi, porcentaje=True)] for r in per.itertuples()]
    tabular("llrrl", ["Asignatura", "Estado en P1", "n", "Reprueban en P3", r"Tasa [IC 95\,\%]"], filas, "p1_persistencia")
    for r in per.itertuples():
        M[f"pers{r.asignatura.title()}{'Rep' if r.estado_p1 == 'reprueba_P1' else 'Apr'}"] = pct(r.tasa)

    # --- P2
    p2 = rd("p2_condiciones")
    filas = []
    for r in p2.itertuples():
        ef = (n(r.valor_efecto) + " " + ic(r.ic_lo, r.ic_hi)) if r.efecto == "δ de Cliff" else \
             (n(100 * r.dif, 1) + " pp " + ic(100 * r.ic_lo, 100 * r.ic_hi, d=1) + r"; $V$=" + n(r.valor_efecto, 3))
        filas.append([ASIG[r.asignatura], tex(r.etiqueta), tex(r.resumen_sobre), tex(r.resumen_bajo), "MW" if r.prueba.startswith("Mann") else r"$\chi^2$", pval(r.p_holm), ef])
    tabular(r"lllllrl", ["Asig.", "Variable", "Sobre prom.", "Bajo prom.", "Prueba", "$p$ Holm", r"Efecto [IC 95\,\%]"], filas, "p2")
    for r in p2.itertuples():
        k = r.asignatura.title() + "".join(w.title() for w in r.variable.replace("sim_", "sim ").replace("_", " ").split())
        M["cd" + k.replace(" ", "")] = n(r.valor_efecto, 3 if r.efecto != "δ de Cliff" else 2)
    nsob = p2.groupby("asignatura").first()
    for a, r in nsob.iterrows():
        M[f"nSobre{a.title()}"] = e(r.n_sobre)
        M[f"nBajo{a.title()}"] = e(r.n_bajo)

    # --- Pregunta principal
    pf = rd("principal_perfiles")
    si = lambda b: "Sí" if b else "No"  # noqa: E731
    filas = [[si(r.alta_inasistencia), si(r.reprob_previas), si(r.bajo_lms_sim), e(r.n), pct(r.tasa, 0) + " " + ic(r.ic_lo, r.ic_hi, porcentaje=True),
              pct(r.sim_pct_con_alerta, 0), n(r.sim_media_tutorias, 2), pct(r.sim_pct_alerta_cerrada, 0)] for r in pf.itertuples()]
    tabular("cccrlrrr", ["Alta inasist.", "Reprob. prev.", "Bajo LMS [SIM]", "n", r"Reprobación [IC 95\,\%]",
                         "Con alerta [SIM]", "Tutorías/matr. [SIM]", "Alerta cerrada [SIM]"], filas, "principal_perfiles")
    rp = pf[pf.reprob_previas]
    M["perfRepMin"], M["perfRepMax"] = pct(rp.tasa.min(), 0), pct(rp.tasa.max(), 0)
    gr = rd("principal_gradiente")
    for r in gr.itertuples():
        M[f"grad{['Cero','Uno','Dos'][r.n_factores_reales]}"] = pct(r.tasa)
        M[f"gradIc{['Cero','Uno','Dos'][r.n_factores_reales]}"] = ic(r.ic_lo, r.ic_hi, porcentaje=True)
        M[f"gradN{['Cero','Uno','Dos'][r.n_factores_reales]}"] = e(r.n)
    orp = pd.read_csv(RESULTS / "principal_or.csv", index_col=0)
    filas = [[tex(r.muestra), e(r.n), tex(i).replace("POR", "Portugués (vs. Matemáticas)"), n(r.OR) + " " + ic(r.ic_lo, r.ic_hi), pval(r.p)]
             for i, r in orp.iterrows()]
    tabular("lrllr", ["Muestra", "n", "Factor", r"OR [IC 95\,\%]", "$p$"], filas, "principal_or")
    pri = orp[orp.muestra.str.contains("principal")]
    M["orInasist"] = n(pri.loc["alta_inasistencia", "OR"])
    M["orInasistIc"] = ic(pri.loc["alta_inasistencia", "ic_lo"], pri.loc["alta_inasistencia", "ic_hi"])
    M["orReprob"] = n(pri.loc["reprob_previas", "OR"])
    M["orReprobIc"] = ic(pri.loc["reprob_previas", "ic_lo"], pri.loc["reprob_previas", "ic_hi"])
    M["nConfiables"] = e(pri.n.iloc[0])
    sen = orp[orp.muestra.str.contains("sensibilidad")]
    M["orInasistTodas"] = n(sen.loc["alta_inasistencia", "OR"])
    M["orInasistTodasIc"] = ic(sen.loc["alta_inasistencia", "ic_lo"], sen.loc["alta_inasistencia", "ic_hi"])

    # --- Modelos
    mm = rd("modelo_metricas")
    orden_esc = ["A_corte_P1", "C_matricula", "C_mas_sim", "A_mas_sim", "A_mas_faltas", "B_corte_P2"]
    nombres = {"A_corte_P1": "A: corte P1 (principal)", "C_matricula": "C: sin notas", "C_mas_sim": "C + LMS [SIM]",
               "A_mas_sim": "A + LMS [SIM]", "A_mas_faltas": "A + faltas anuales", "B_corte_P2": "B: corte P2"}
    filas = []
    for es in orden_esc:
        for r in mm[mm.escenario == es].itertuples():
            filas.append([nombres[es], tex(r.modelo), n(r.cv_auc_roc, 3) + r" $\pm$ " + n(r.cv_auc_roc_de, 3), n(r.cv_auc_pr, 3),
                          n(r.auc_roc, 3) + " " + ic(r.auc_ic_lo, r.auc_ic_hi, 3), n(r.auc_pr, 3), n(r.recall, 3), n(r.precision, 3),
                          n(r.f1, 3), n(r.brier, 3)])
    tabular("llllllllll", ["Escenario", "Modelo", "AUC-ROC CV", "AUC-PR CV", r"AUC-ROC test [IC 95\,\%]", "AUC-PR", "Recall",
                           "Precisión", "F1", "Brier"], filas, "modelos")
    key = {"Línea base (prior)": "Base", "Regresión logística": "Lr", "Random forest": "Rf", "Gradient boosting": "Gb"}
    kesc = {"A_corte_P1": "A", "B_corte_P2": "B", "C_matricula": "C", "C_mas_sim": "Csim", "A_mas_sim": "Asim", "A_mas_faltas": "Afal"}
    for r in mm.itertuples():
        p = f"{kesc[r.escenario]}{key[r.modelo]}"
        M[f"auc{p}"] = n(r.auc_roc, 3)
        M[f"aucIc{p}"] = ic(r.auc_ic_lo, r.auc_ic_hi, 3)
        M[f"aucpr{p}"] = n(r.auc_pr, 3)
        M[f"rec{p}"] = n(r.recall, 3)
        M[f"prec{p}"] = n(r.precision, 3)
        M[f"fone{p}"] = n(r.f1, 3)
        M[f"brier{p}"] = n(r.brier, 3)
        M[f"esp{p}"] = n(r.especificidad, 3)
        M[f"cvauc{p}"] = n(r.cv_auc_roc, 3)
        M[f"umbral{p}"] = n(r.umbral, 3)
        M[f"tp{p}"], M[f"fp{p}"], M[f"fn{p}"], M[f"tn{p}"] = e(r.tp), e(r.fp), e(r.fn), e(r.tn)
    r0 = mm[mm.escenario == "A_corte_P1"].iloc[0]
    M["prevTest"] = pct((r0.tp + r0.fn) / (r0.tp + r0.fn + r0.fp + r0.tn))
    M["nRiesgoTest"] = e(r0.tp + r0.fn)
    sens = rd("modelo_sensibilidad")
    filas = [[tex(r.modelo), tex(r.comparacion.replace("_", " ")), n(r.delta_auc, 3) + " " + ic(r.ic_lo, r.ic_hi, 3)] for r in sens.itertuples()]
    tabular("lll", ["Modelo", "Comparación", r"$\Delta$ AUC-ROC test [IC 95\,\% bootstrap pareado]"], filas, "sensibilidad")
    for r in sens.itertuples():
        a, b = r.comparacion.split(" − ")
        M[f"d{kesc[a]}{kesc[b]}{key[r.modelo]}"] = n(r.delta_auc, 3)
        M[f"dIc{kesc[a]}{kesc[b]}{key[r.modelo]}"] = ic(r.ic_lo, r.ic_hi, 3)
    circ = rd("modelo_contrafactual_circular").iloc[0]
    M["circDifC"] = n(circ["auc_C_mas_circular"] - circ["auc_C_honesto"], 3)
    for k, v in circ.items():
        M["circ" + "".join(w.title() for w in k.replace("auc_", "").split("_"))] = n(v, 3)
    orm = pd.read_csv(RESULTS / "modelo_odds_ratios.csv", index_col=0)
    etiq = {"nota_p1": "Nota P1 (G1), por punto", "reprobaciones_previas": "Reprobaciones previas (por unidad)",
            "tiempo_estudio": "Tiempo de estudio (por nivel)", "edad": "Edad (por año)", "salidas": "Salidas con amigos (1–5)",
            "alcohol_finde": "Alcohol fin de semana (1–5)", "asignatura_POR": "Portugués (vs. Matemáticas)",
            "escuela_MS": "Mousinho da Silveira (vs. GP)", "sexo_M": "Sexo masculino", "desea_superior": "Aspira a educación superior",
            "apoyo_escolar": "Refuerzo escolar", "zona_rural": "Zona rural"}
    filas = [[tex(etiq[i]), n(r.OR) + " " + ic(r.ic_lo, r.ic_hi), pval(r.p)] for i, r in orm.iterrows()]
    tabular("lll", ["Predictor (corte P1)", r"OR [IC 95\,\%]", "$p$"], filas, "odds")
    M["orNotaUno"] = n(orm.loc["nota_p1", "OR"], 3)
    M["orNotaUnoIc"] = ic(orm.loc["nota_p1", "ic_lo"], orm.loc["nota_p1", "ic_hi"], 3)
    M["orNotaUnoRed"] = n(100 * (1 - orm.loc["nota_p1", "OR"]), 0) + r"\,\%"
    M["orPor"] = n(orm.loc["asignatura_POR", "OR"])
    M["orPorIc"] = ic(orm.loc["asignatura_POR", "ic_lo"], orm.loc["asignatura_POR", "ic_hi"])
    M["orSup"] = n(orm.loc["desea_superior", "OR"])
    M["orSupIc"] = ic(orm.loc["desea_superior", "ic_lo"], orm.loc["desea_superior", "ic_hi"])
    M["orEsc"] = n(orm.loc["escuela_MS", "OR"])
    M["orEscIc"] = ic(orm.loc["escuela_MS", "ic_lo"], orm.loc["escuela_MS", "ic_hi"])
    imp = rd("modelo_importancia_permutacion")
    M["impNotaUno"] = n(imp.iloc[0].importancia, 3)
    M["impSegunda"] = tex(imp.iloc[1].variable)
    M["impSegundaV"] = n(imp.iloc[1].importancia, 3)
    est = rd("modelo_estratos")
    filas = [[tex(r.estrato), tex(ESC.get(r.valor, ASIG.get(r.valor, r.valor))), e(r.n), pct(r.prevalencia), n(r.auc_roc, 3), n(r.recall, 3), n(r.precision, 3)]
             for r in est.itertuples()]
    tabular("llrrrrr", ["Estrato", "Valor", "n test", "Prevalencia", "AUC-ROC", "Recall", "Precisión"], filas, "estratos")
    resumen = json.loads((RESULTS / "resumen_analisis.json").read_text(encoding="utf-8"))
    M["mejorNl"] = tex(resumen["mejor_no_lineal"])
    M["exactitudTexto"] = pct(resumen["texto"]["exactitud"])
    M["nNotasTexto"] = e(resumen["texto"]["n_notas"])

    # --- Texto
    term = rd("texto_terminos").head(10)
    filas = [[tex(r.termino), e(r.frecuencia)] for r in term.itertuples()]
    tabular("lr", ["Término", "Frecuencia"], filas, "texto_terminos")
    ej = rd("texto_ejemplos").head(3)
    filas = [[r"\footnotesize " + tex(r.texto), r"\footnotesize\texttt{" + tex(r.norm) + "}", tex(r.cat)] for r in ej.itertuples()]
    tabular("p{5.6cm}p{5.6cm}l", ["Texto original [SIM]", "Texto normalizado", "Categoría"], filas, "texto_ejemplos")

    # --- Escalabilidad
    sc = rd("escalabilidad")
    filas = [[r"$\times$" + e(r.factor), e(r.filas), e(r.grupos), n(r.memoria_bytes / 1e6, 1), n(r.parquet_bytes / 1e6, 1),
              n(r.t_agregacion_s, 3), n(r.t_escritura_s, 3), n(r.t_lectura_s, 3)] for r in sc.itertuples()]
    tabular("rrrrrrrr", ["Escala", "Eventos", "Grupos", "Memoria (MB)", "Parquet (MB)", "Agregación (s)", "Escritura (s)", "Lectura (s)"],
            filas, "escalabilidad")
    se = rd("escalabilidad_escenarios")
    filas = [[tex(r.escenario), e(r.eventos_anuales), n(r.memoria_gb, 2), n(r.t_agregacion_estimado_s, 1), "Sí" if r.cabe_en_un_nodo_8gb else "No"]
             for r in se.itertuples()]
    tabular("p{6.2cm}rrrc", ["Escenario (supuesto)", "Eventos/año", "Memoria (GB)", "Agregación estimada (s)", "Un nodo 8 GB"], filas, "escenarios")
    rs = json.loads((RESULTS / "escalabilidad_resumen.json").read_text(encoding="utf-8"))
    M.update(benchEquipo=tex(rs["equipo"]), benchBpf=n(rs["bytes_por_fila"], 1), benchFps=n(rs["filas_por_s"] / 1e6, 1),
             benchMaxNodo=n(rs["filas_max_nodo_8gb"] / 1e6, 0), benchMongo=n(rs["mongo_ag01_mediana_s"], 3),
             benchEvMat=n(rs["eventos_por_matricula_anio"], 1), benchUciKb=n(rs["dataset_uci_bytes"] / 1e3, 0),
             benchMaxFilas=e(sc.filas.max()), benchMaxT=n(sc.t_agregacion_s.max(), 2), benchMaxMem=n(sc.memoria_bytes.max() / 1e6, 0),
             benchMaxFactor=e(sc.factor.max()),
             benchOrdenes=n(np.log10(rs["filas_max_nodo_8gb"] / sc.filas.iloc[0]), 1))

    # --- Diccionario, linaje, mapeo avance, identidad
    dic = pd.read_csv(DATA_PROCESSED / "diccionario_datos.csv")
    filas = [[r"\texttt{" + tex(r.variable) + "}", tex(r.tipo), tex(r.dominio), tex(r.origen), tex(r.descripcion)] for r in dic.itertuples()]
    tabular(r"p{3.9cm}p{1.2cm}p{2.2cm}p{1.9cm}p{5.6cm}", ["Variable", "Tipo", "Dominio", "Origen", "Descripción"], filas, "diccionario")
    longtable(r">{\raggedright\arraybackslash}p{3.9cm}p{1.2cm}>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}p{4.8cm}", ["Variable", "Tipo", "Dominio", "Origen", "Descripción"], filas,
              "diccionario_lt", "Diccionario de datos del dataset analítico.", "tab:diccionario")
    filas = [[r"\texttt{" + tex(r.variable) + "}", tex(r.origen), tex(r.fuente_campo), tex(r.regla)] for r in dic.itertuples()]
    tabular(r"p{3.9cm}p{2.0cm}p{6.8cm}p{2.2cm}", ["Variable", "Origen", "Tabla/colección y campo (transformación)", "Regla"], filas, "linaje")
    longtable(r">{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}p{6.2cm}>{\raggedright\arraybackslash}p{2.1cm}", ["Variable", "Origen", "Tabla/colección y campo (transformación)", "Regla"], filas,
              "linaje_lt", "Linaje: de cada variable a su tabla/colección, campo y regla de transformación.", "tab:linaje")
    ej_all = json.loads((RESULTS / "mongo_documentos_ejemplo.json").read_text(encoding="utf-8"))
    (GENERATED_TEX / "ejemplo_seguimiento.json").write_text(json.dumps(ej_all["seguimiento_riesgo"], ensure_ascii=False, indent=1), encoding="utf-8")
    ej_doc = ej_all["alertas_riesgo"]
    (GENERATED_TEX / "ejemplo_alerta.json").write_text(json.dumps(ej_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    mp = pd.read_csv(DATA_PROCESSED / "mapeo_avance_final.csv")
    filas = [[r"\texttt{" + tex(r.variable_avance) + "}", r"\texttt{" + tex(r.variable_final) + "}", tex(r.decision)] for r in mp.itertuples()]
    tabular(r"p{3.6cm}p{4.4cm}p{6.8cm}", ["Variable del avance", "Variable final", "Decisión"], filas, "mapeo_avance")
    fn = pd.read_csv(DATA_STAGING / "identidad_posibles_falsos_negativos.csv")
    filas = [[e(r.fila_mat), e(r.fila_por), tex(r.atributo_distinto), tex(r.valor_mat), tex(r.valor_por)] for r in fn.itertuples()]
    tabular("rrlrr", ["Línea MAT", "Línea POR", "Atributo distinto", "Valor MAT", "Valor POR"], filas, "falsos_negativos")
    M["nFalsosNeg"] = e(len(fn))
    M["nEstMin"] = e(idn["estudiantes"] - len(fn))

    # --- Checksums
    art = log["artefactos"]
    sel = [k for k in art if k.startswith("data/processed") or k.endswith("uci_raw.parquet")]
    filas = [[r"\texttt{" + tex(k) + "}", e(art[k]["filas"]) if art[k]["filas"] else "--", r"\texttt{" + art[k]["sha256"][:16] + "…}"] for k in sel]
    filas += [[r"\texttt{data/raw/student-" + a.lower() + ".csv}", e(et[f"extract_{a}"]["filas_leidas"]),
               r"\texttt{" + et[f"extract_{a}"]["sha256"][:16] + "…}"] for a in ("MAT", "POR")]
    tabular("lrl", ["Artefacto", "Filas", "SHA-256 (prefijo)"], filas, "checksums")

    with open(GENERATED_TEX / "valores.tex", "w", encoding="utf-8") as f:
        f.write("% Generado automáticamente por src/report_tables.py — NO EDITAR\n")
        palabras = dict(zip("0123456789", ["Cero", "Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve"]))
        M2 = {"".join(palabras.get(c, c) for c in k): v for k, v in M.items()}
        for k, v in sorted(M2.items()):
            assert k.isalpha(), f"Nombre de macro inválido: {k}"
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    run()
