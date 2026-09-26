"""ANALYSIS — respuestas a P1, P2, pregunta principal y modelo de predicción temprana.

Entrada : data/processed/dataset_analitico.parquet
Salidas : results/*.csv|json (tablas y métricas) e informe/figuras/*.png

Reglas contra la fuga de información (leakage):
  L-01  G3 (nota_p3) nunca es predictor: solo define en_riesgo.
  L-02  `faltas` es un total ANUAL: se excluye del modelo temprano (solo sensibilidad).
  L-03  Variables simuladas del corte P1: solo eventos LMS con fecha <= fin de P1.
        Tutorías y alertas (P2–P3) son posteriores al corte: solo descriptivas.
  L-04  Partición train/test agrupada por estudiante (fijada en integrate.py).
Los hallazgos sobre datos reales (UCI) y las demostraciones con datos simulados se
reportan por separado; toda tabla marca las variables `sim_`.
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from config import DATA_PROCESSED, DATA_STAGING, DATA_SYNTH_RAW, FIGURES, RESULTS, SEED, ensure_dirs  # noqa: E402
from tracking import etapa, get_logger  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
N_BOOT = 2000
RECALL_OBJETIVO = 0.80  # prioridad de alerta temprana: detectar >= 80 % de los casos

# Paleta categórica de referencia (3 primeros slots, validados en todos los pares)
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, TINTA, TINTA2 = "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})
ASIG = {"MAT": "Matemáticas", "POR": "Lengua Portuguesa"}
ESC = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}


# --------------------------------------------------------------------------- #
# Utilidades estadísticas
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int) -> tuple[float, float]:
    """IC 95 % de Wilson para una proporción."""
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Delta de Cliff: P(X > Y) − P(X < Y)."""
    y = np.sort(y)
    mayor = np.searchsorted(y, x, side="left").sum()
    menor = len(y) - np.searchsorted(y, x, side="right")
    return float((mayor - menor.sum()) / (len(x) * len(y)))


def boot_ci(fun, x, y, rng, n=N_BOOT) -> tuple[float, float]:
    """IC 95 % percentil bootstrap para un estadístico de dos muestras."""
    vals = [fun(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cramer_v(tabla: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.sum()
    return float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))


def mag_cliff(d: float) -> str:
    """Magnitud según Romano et al. (2006)."""
    a = abs(d)
    return "despreciable" if a < 0.147 else "pequeño" if a < 0.33 else "mediano" if a < 0.474 else "grande"


# --------------------------------------------------------------------------- #
# Descriptivos
# --------------------------------------------------------------------------- #
def descriptivos(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for a, g in df.groupby("asignatura", observed=True):
        k, n = int(g.en_riesgo.sum()), len(g)
        lo, hi = wilson(k, n)
        q = lambda s: f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"  # noqa: E731
        filas.append({"asignatura": a, "n_matriculas": n, "en_riesgo": k, "prevalencia": k / n,
                      "ic_lo": lo, "ic_hi": hi,
                      "nota_p1_mediana_ric": q(g.nota_p1[~g.no_evaluado_p1]),
                      "nota_p3_mediana_ric": q(g.nota_p3[~g.no_evaluado_p3]),
                      "faltas_mediana_ric": q(g.faltas), "no_evaluados_p3": int(g.no_evaluado_p3.sum()),
                      "con_reprob_previas": int((g.reprobaciones_previas > 0).sum())})
    k, n = int(df.en_riesgo.sum()), len(df)
    lo, hi = wilson(k, n)
    filas.append({"asignatura": "Total", "n_matriculas": n, "en_riesgo": k, "prevalencia": k / n,
                  "ic_lo": lo, "ic_hi": hi, "nota_p1_mediana_ric": "", "nota_p3_mediana_ric": "",
                  "faltas_mediana_ric": "", "no_evaluados_p3": int(df.no_evaluado_p3.sum()),
                  "con_reprob_previas": int((df.reprobaciones_previas > 0).sum())})
    out = pd.DataFrame(filas)
    out.to_csv(RESULTS / "descriptivos.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# P1 — evolución de la nota media y reprobación por centro × asignatura
# --------------------------------------------------------------------------- #
def p1(df: pd.DataFrame, rng) -> dict:
    largo = []
    for p, (col, ne) in {"P1": ("nota_p1", "no_evaluado_p1"), "P2": ("nota_p2", "no_evaluado_p2"),
                         "P3": ("nota_p3", "no_evaluado_p3")}.items():
        t = df[["estudiante_id", "asignatura", "escuela", col, ne]].rename(columns={col: "nota", ne: "no_ev"})
        t["periodo"] = p
        largo.append(t)
    largo = pd.concat(largo)
    ev = largo[~largo.no_ev]

    def resumen(g):
        n, m, s = len(g), g.nota.mean(), g.nota.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
        return pd.Series({"n": n, "media": m, "de": s, "ic_lo": m - h, "ic_hi": m + h})
    medias = ev.groupby(["asignatura", "periodo"], observed=True).apply(resumen).reset_index()
    medias_esc = ev.groupby(["asignatura", "escuela", "periodo"], observed=True).apply(resumen).reset_index()
    # Sensibilidad: medias SIN excluir los ceros (lo que haría un AVG ingenuo)
    ing = largo.groupby(["asignatura", "periodo"], observed=True).nota.mean().rename("media_con_ceros")
    medias = medias.merge(ing.reset_index(), on=["asignatura", "periodo"])
    medias.to_csv(RESULTS / "p1_medias_periodo.csv", index=False)
    medias_esc.to_csv(RESULTS / "p1_medias_periodo_escuela.csv", index=False)

    # Friedman (medidas repetidas) + W de Kendall + Wilcoxon post hoc con Holm
    pruebas = []
    for a, g in df.groupby("asignatura", observed=True):
        c = g[~(g.no_evaluado_p1 | g.no_evaluado_p2 | g.no_evaluado_p3)]
        fr = stats.friedmanchisquare(c.nota_p1, c.nota_p2, c.nota_p3)
        W = fr.statistic / (len(c) * 2)
        pares = [("nota_p1", "nota_p2"), ("nota_p2", "nota_p3"), ("nota_p1", "nota_p3")]
        pv, filas = [], []
        for x, y in pares:
            d = c[y] - c[x]
            w = stats.wilcoxon(c[x], c[y], zero_method="wilcox")
            dn = d[d != 0]
            r = stats.rankdata(np.abs(dn))
            rrb = (r[dn > 0].sum() - r[dn < 0].sum()) / r.sum()  # correlación biserial de rangos
            rb = np.random.default_rng(SEED)
            b = [rb.choice(d.values, len(d)).mean() for _ in range(N_BOOT)]
            filas.append({"asignatura": a, "comparacion": f"{x[-2:].upper()}→{y[-2:].upper()}",
                          "n": len(c), "dif_media": d.mean(), "dif_ic_lo": np.percentile(b, 2.5),
                          "dif_ic_hi": np.percentile(b, 97.5), "p": w.pvalue, "r_rb": rrb})
            pv.append(w.pvalue)
        adj = multipletests(pv, method="holm")[1]
        for f, pa in zip(filas, adj):
            f.update(p_holm=pa, friedman_chi2=fr.statistic, friedman_p=fr.pvalue, kendall_w=W)
        pruebas += filas
    pruebas = pd.DataFrame(pruebas)
    pruebas.to_csv(RESULTS / "p1_pruebas_evolucion.csv", index=False)

    # Tasa de reprobación final por escuela × asignatura (Wilson) y chi-cuadrado
    tasa_global = df.en_riesgo.mean()
    rep = []
    for (a, e), g in df.groupby(["asignatura", "escuela"], observed=True):
        k, n = int(g.en_riesgo.sum()), len(g)
        lo, hi = wilson(k, n)
        rep.append({"asignatura": a, "escuela": e, "n": n, "reprobados": k, "tasa": k / n,
                    "ic_lo": lo, "ic_hi": hi, "prioridad": lo > tasa_global})
    rep = pd.DataFrame(rep)
    chi = []
    for a, g in df.groupby("asignatura", observed=True):
        tab = pd.crosstab(g.escuela, g.en_riesgo).values
        c2 = stats.chi2_contingency(tab, correction=False)
        rr = g.groupby("escuela", observed=True).en_riesgo.mean()
        chi.append({"asignatura": a, "chi2": c2[0], "p": c2[1], "v_cramer": cramer_v(tab),
                    "dif_tasas_MS_GP": rr.get("MS", np.nan) - rr.get("GP", np.nan)})
    chi = pd.DataFrame(chi)
    chi["p_holm"] = multipletests(chi.p, method="holm")[1]
    rep.to_csv(RESULTS / "p1_reprobacion_escuela_asignatura.csv", index=False)
    chi.to_csv(RESULTS / "p1_chi2_escuela.csv", index=False)

    # Persistencia: P(reprueba P3 | reprueba P1)
    trans = []
    for a, g in df.groupby("asignatura", observed=True):
        for estado, s in (("reprueba_P1", g.nota_p1 < 10), ("aprueba_P1", g.nota_p1 >= 10)):
            k, n = int(g[s].en_riesgo.sum()), int(s.sum())
            lo, hi = wilson(k, n)
            trans.append({"asignatura": a, "estado_p1": estado, "n": n, "reprueban_p3": k,
                          "tasa": k / n, "ic_lo": lo, "ic_hi": hi})
    trans = pd.DataFrame(trans)
    trans.to_csv(RESULTS / "p1_persistencia.csv", index=False)

    # Figura 1: evolución de medias con IC 95 % por escuela
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, a in zip(axs, ["MAT", "POR"]):
        for e, col in (("GP", AZUL), ("MS", NARANJA)):
            s = medias_esc[(medias_esc.asignatura == a) & (medias_esc.escuela == e)]
            x = np.arange(3) + (-0.04 if e == "GP" else 0.04)
            ax.errorbar(x, s.media, yerr=[s.media - s.ic_lo, s.ic_hi - s.media], color=col,
                        marker="o", ms=5, capsize=3, lw=2, label=ESC[e])
            ax.annotate(f"{s.media.iloc[-1]:.1f}", (x[-1], s.media.iloc[-1]), xytext=(11, 0),
                        textcoords="offset points", va="center", color=TINTA2, fontsize=8)
        ax.set_xticks(range(3), ["P1 (G1)", "P2 (G2)", "P3 (G3)"])
        ax.set_title(ASIG[a], loc="left", color=TINTA)
        ax.axhline(10, color=GRIS, lw=1, ls="--")
        ax.text(2.25, 10.1, "umbral 10", color=GRIS, fontsize=7, ha="right", va="bottom")
    axs[0].set_ylabel("Nota media (0–20), evaluados")
    axs[0].set_ylim(9, 14)
    axs[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_evolucion.png")
    plt.close(fig)

    # Figura 2: tasas de reprobación con IC de Wilson
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    etiquetas = [f"{ASIG[r.asignatura]}\n{ESC[r.escuela]}" for r in rep.itertuples()]
    y = np.arange(len(rep))
    cols = [AZUL if r.escuela == "GP" else NARANJA for r in rep.itertuples()]
    ax.hlines(y, rep.ic_lo * 100, rep.ic_hi * 100, color=cols, lw=2)
    ax.scatter(rep.tasa * 100, y, color=cols, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    for yi, r in zip(y, rep.itertuples()):
        ax.text(max(r.ic_hi, tasa_global) * 100 + 1.5, yi, f"{r.tasa*100:.1f} % (n={r.n})", va="center", fontsize=8, color=TINTA2)
    ax.axvline(tasa_global * 100, color=GRIS, ls="--", lw=1)
    ax.text(tasa_global * 100 + 0.5, -0.45, f"tasa global {tasa_global*100:.1f} %", color=GRIS,
            fontsize=7, ha="left", va="center")
    ax.set_ylim(-0.7, len(rep) - 0.5)
    ax.set_yticks(y, etiquetas, fontsize=8)
    ax.set_xlabel("Tasa de reprobación final (G3 < 10), % con IC 95 % de Wilson")
    ax.set_xlim(0, 65)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_reprobacion.png")
    plt.close(fig)
    return {"medias": medias, "pruebas": pruebas, "reprobacion": rep, "chi2": chi, "persistencia": trans}


# --------------------------------------------------------------------------- #
# P2 — condiciones que distinguen sobre/bajo el promedio
# --------------------------------------------------------------------------- #
VARS_P2 = [  # variable, tipo, etiqueta, simulada
    ("tiempo_estudio", "ord", "Tiempo de estudio (1–4)", False),
    ("faltas", "num", "Faltas anuales", False),
    ("reprobaciones_previas", "ord", "Reprobaciones previas", False),
    ("apoyo_escolar", "bin", "Refuerzo escolar (sí)", False),
    ("clases_pagadas", "bin", "Clases pagadas (sí)", False),
    ("desea_superior", "bin", "Aspira a educación superior (sí)", False),
    ("sim_lms_minutos_p1", "num", "Minutos LMS en P1 [SIM]", True),
    ("sim_lms_eventos_p1", "num", "Eventos LMS en P1 [SIM]", True),
]


def p2(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Compara sobre vs. bajo el promedio (evaluados en P3), por asignatura."""
    filas = []
    for a, g in df[~df.no_evaluado_p3].groupby("asignatura", observed=True):
        sob = g[g.rendimiento_sobre_promedio == True]  # noqa: E712
        baj = g[g.rendimiento_sobre_promedio == False]  # noqa: E712
        for v, tipo, et, sim in VARS_P2:
            x, y = sob[v].astype(float).values, baj[v].astype(float).values
            f = {"asignatura": a, "variable": v, "etiqueta": et, "simulada": sim,
                 "n_sobre": len(x), "n_bajo": len(y)}
            if tipo == "bin":
                tab = np.array([[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]])
                c2 = stats.chi2_contingency(tab, correction=False)
                f.update(prueba="Chi-cuadrado", estadistico=c2[0], p=c2[1],
                         resumen_sobre=f"{x.mean()*100:.1f} %", resumen_bajo=f"{y.mean()*100:.1f} %",
                         efecto="V de Cramér", valor_efecto=cramer_v(tab))
                rb = np.random.default_rng(SEED)
                b = [rb.choice(x, len(x)).mean() - rb.choice(y, len(y)).mean() for _ in range(N_BOOT)]
                f.update(ic_lo=np.percentile(b, 2.5), ic_hi=np.percentile(b, 97.5),
                         dif=x.mean() - y.mean(), magnitud="")
            else:
                mw = stats.mannwhitneyu(x, y, alternative="two-sided")
                d = cliff_delta(x, y)
                lo, hi = boot_ci(cliff_delta, x, y, np.random.default_rng(SEED))
                f.update(prueba="Mann-Whitney U", estadistico=mw.statistic, p=mw.pvalue,
                         resumen_sobre=f"{np.median(x):.0f} [{np.percentile(x,25):.0f}–{np.percentile(x,75):.0f}]",
                         resumen_bajo=f"{np.median(y):.0f} [{np.percentile(y,25):.0f}–{np.percentile(y,75):.0f}]",
                         efecto="δ de Cliff", valor_efecto=d, ic_lo=lo, ic_hi=hi, dif=d,
                         magnitud=mag_cliff(d))
            filas.append(f)
    out = pd.DataFrame(filas)
    out["p_holm"] = np.nan
    for a in out.asignatura.unique():
        m = out.asignatura == a
        out.loc[m, "p_holm"] = multipletests(out.loc[m, "p"], method="holm")[1]
    out.to_csv(RESULTS / "p2_condiciones.csv", index=False)

    # Figura 3: forest plot de delta de Cliff (variables ordinales/numéricas)
    num = out[out.efecto == "δ de Cliff"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, a in zip(axs, ["MAT", "POR"]):
        s = num[num.asignatura == a].reset_index(drop=True)
        y = np.arange(len(s))[::-1]
        cols = [NARANJA if r.simulada else AZUL for r in s.itertuples()]
        ax.hlines(y, s.ic_lo, s.ic_hi, color=cols, lw=2)
        ax.scatter(s.valor_efecto, y, color=cols, s=36, zorder=3, edgecolor="white", linewidth=1.5)
        ax.axvline(0, color=GRIS, lw=1)
        ax.set_title(ASIG[a], loc="left", color=TINTA)
        ax.set_yticks(y, s.etiqueta, fontsize=8)
        ax.set_xlim(-0.6, 0.6)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("δ de Cliff (grupo sobre el promedio − grupo bajo el promedio), IC 95 % bootstrap",
                  fontsize=8.5, color=TINTA2)
    axs[0].scatter([], [], color=AZUL, label="dato real (UCI)")
    axs[0].scatter([], [], color=NARANJA, label="dato simulado")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "fig_p2_cliff.png")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Pregunta principal — combinación de factores y tutorías/alertas asociadas
# --------------------------------------------------------------------------- #
def principal(df: pd.DataFrame) -> dict:
    d = df.copy()
    q75 = d.groupby("asignatura", observed=True).faltas.transform(lambda s: s.quantile(0.75))
    q25 = d.groupby("asignatura", observed=True).sim_lms_minutos_p1.transform(lambda s: s.quantile(0.25))
    d["f_inasistencia"] = d.faltas >= q75                 # real: cuartil superior de faltas
    d["f_reprob_previas"] = d.reprobaciones_previas >= 1  # real
    d["f_bajo_lms"] = d.sim_lms_minutos_p1 <= q25         # SIMULADO: cuartil inferior LMS P1
    d["n_factores_reales"] = d.f_inasistencia.astype(int) + d.f_reprob_previas.astype(int)
    d["n_factores_total"] = d.n_factores_reales + d.f_bajo_lms.astype(int)

    filas = []
    for (fi, fr, fl), g in d[d.faltas_confiable].groupby(["f_inasistencia", "f_reprob_previas", "f_bajo_lms"]):
        k, n = int(g.en_riesgo.sum()), len(g)
        lo, hi = wilson(k, n)
        filas.append({"alta_inasistencia": fi, "reprob_previas": fr, "bajo_lms_sim": fl, "n": n,
                      "reprobados": k, "tasa": k / n, "ic_lo": lo, "ic_hi": hi,
                      "sim_pct_con_alerta": (g.sim_n_alertas > 0).mean(),
                      "sim_media_tutorias": g.sim_n_tutorias.mean(),
                      "sim_pct_con_tutoria": (g.sim_n_tutorias > 0).mean(),
                      "sim_pct_alerta_cerrada": (g.sim_alerta_cerrada.astype(bool)).mean()})
    perfiles = pd.DataFrame(filas).sort_values("tasa", ascending=False)
    perfiles.to_csv(RESULTS / "principal_perfiles.csv", index=False)

    grad = []
    for n_f, g in d[d.faltas_confiable].groupby("n_factores_reales"):
        k, n = int(g.en_riesgo.sum()), len(g)
        lo, hi = wilson(k, n)
        grad.append({"n_factores_reales": n_f, "n": n, "reprobados": k, "tasa": k / n, "ic_lo": lo, "ic_hi": hi})
    grad = pd.DataFrame(grad)
    grad.to_csv(RESULTS / "principal_gradiente.csv", index=False)

    # Regresión logística con errores agrupados por estudiante: OR por factor real.
    # Análisis principal sobre faltas confiables (R-09); sensibilidad con todas.
    def _or(dd, etiqueta):
        X = sm.add_constant(pd.DataFrame({
            "alta_inasistencia": dd.f_inasistencia.astype(float),
            "reprob_previas": dd.f_reprob_previas.astype(float),
            "POR": (dd.asignatura == "POR").astype(float)}))
        m = sm.GLM(dd.en_riesgo.astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": dd.estudiante_id.values})
        o = pd.DataFrame({"OR": np.exp(m.params), "ic_lo": np.exp(m.conf_int()[0]),
                          "ic_hi": np.exp(m.conf_int()[1]), "p": m.pvalues}).drop("const")
        o["muestra"], o["n"] = etiqueta, len(dd)
        return o
    ors = pd.concat([_or(d[d.faltas_confiable], "faltas confiables (principal)"),
                     _or(d, "todas las matrículas (sensibilidad)")])
    ors.to_csv(RESULTS / "principal_or.csv")

    # Figura 4
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(grad.n_factores_reales, grad.tasa * 100, color=AZUL, width=0.55)
    ax.errorbar(grad.n_factores_reales, grad.tasa * 100,
                yerr=[(grad.tasa - grad.ic_lo) * 100, (grad.ic_hi - grad.tasa) * 100],
                fmt="none", ecolor=TINTA2, capsize=3, lw=1)
    for r in grad.itertuples():
        ax.text(r.n_factores_reales, r.ic_hi * 100 + 2, f"{r.tasa*100:.0f} %\n(n={r.n})",
                ha="center", fontsize=7.5, color=TINTA2)
    ax.set_xticks([0, 1, 2], ["0", "1", "2"])
    ax.set_xlabel("Nº de factores reales (alta inasistencia, reprob. previas)\nmatrículas con faltas confiables")
    ax.set_ylabel("Reprobación final (%)")
    ax.set_ylim(0, 85)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_principal_factores.png")
    plt.close(fig)
    return {"perfiles": perfiles, "gradiente": grad, "or": ors, "df": d}


# --------------------------------------------------------------------------- #
# Modelo de predicción temprana
# --------------------------------------------------------------------------- #
CAT = ["asignatura", "escuela", "sexo", "zona", "tamano_familia", "estado_padres", "trabajo_madre",
       "trabajo_padre", "motivo_eleccion", "tutor_legal"]
BIN = ["apoyo_escolar", "apoyo_familiar", "actividades", "guarderia", "desea_superior", "internet",
       "relacion_romantica", "clases_pagadas"]
NUM = ["edad", "educ_madre", "educ_padre", "tiempo_viaje", "tiempo_estudio", "relacion_familiar",
       "tiempo_libre", "salidas", "alcohol_semana", "alcohol_finde", "salud", "reprobaciones_previas"]
SIM_P1 = ["sim_lms_eventos_p1", "sim_lms_minutos_p1", "sim_lms_dias_activos_p1", "sim_lms_tareas_revisadas_p1"]
ESCENARIOS = {
    "C_matricula": {"num": NUM, "bin": BIN, "desc": "Antecedentes al matricular (sin notas)"},
    "A_corte_P1": {"num": NUM + ["nota_p1"], "bin": BIN + ["no_evaluado_p1"], "desc": "Corte P1: + G1 (principal)"},
    "B_corte_P2": {"num": NUM + ["nota_p1", "nota_p2"], "bin": BIN + ["no_evaluado_p1", "no_evaluado_p2"],
                   "desc": "Corte P2: + G2 (secundario)"},
    "A_mas_sim": {"num": NUM + ["nota_p1"] + SIM_P1, "bin": BIN + ["no_evaluado_p1"],
                  "desc": "Corte P1 + LMS simulado (sensibilidad)"},
    "A_mas_faltas": {"num": NUM + ["nota_p1", "faltas"], "bin": BIN + ["no_evaluado_p1"],
                     "desc": "Corte P1 + faltas anuales (optimista, L-02)"},
    "C_mas_sim": {"num": NUM + SIM_P1, "bin": BIN, "desc": "Sin notas + LMS simulado (sensibilidad)"},
}


def modelos() -> dict:
    return {
        "Línea base (prior)": DummyClassifier(strategy="prior"),
        "Regresión logística": LogisticRegression(C=1.0, max_iter=2000),
        "Random forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                random_state=SEED, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200,
                                                            l2_regularization=1.0, random_state=SEED),
    }


def _pipe(esc: dict, clf) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CAT),
        ("num", StandardScaler(), esc["num"] + esc["bin"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def _X(df, esc):
    X = df[CAT + esc["num"] + esc["bin"]].copy()
    for c in esc["bin"]:
        X[c] = X[c].astype(int)
    for c in CAT:
        X[c] = X[c].astype(str)
    return X


def _metricas(y, p, umbral) -> dict:
    yhat = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p),
            "recall": recall_score(y, yhat, zero_division=0),
            "precision": precision_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0), "especificidad": tn / (tn + fp) if tn + fp else np.nan,
            "brier": brier_score_loss(y, p), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def _umbral_recall(y, p, objetivo=RECALL_OBJETIVO) -> float:
    """Mayor umbral cuyo recall en predicciones OOF de entrenamiento es >= objetivo."""
    for u in np.sort(np.unique(p))[::-1]:
        if recall_score(y, (p >= u).astype(int)) >= objetivo:
            return float(u)
    return float(np.min(p))


def _boot_grupos(y, p, grupos, rng, fun=roc_auc_score, p2=None, n=N_BOOT):
    """Bootstrap por estudiante (conglomerado) de una métrica o de la diferencia de dos modelos."""
    ug = np.unique(grupos)
    idx_g = {g: np.where(grupos == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_g[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[idx])) < 2:
            continue
        v = fun(y[idx], p[idx])
        if p2 is not None:
            v = v - fun(y[idx], p2[idx])
        vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def modelado(df: pd.DataFrame, rng) -> dict:
    log = get_logger()
    tr, te = df[df.particion == "train"].reset_index(drop=True), df[df.particion == "test"].reset_index(drop=True)
    ytr, yte = tr.en_riesgo.values, te.en_riesgo.values
    gtr, gte = tr.estudiante_id.values, te.estudiante_id.values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(tr, ytr, groups=gtr))
    filas, preds, ajustados = [], {}, {}
    for en, esc in ESCENARIOS.items():
        Xtr, Xte = _X(tr, esc), _X(te, esc)
        for mn, clf in modelos().items():
            if en != "A_corte_P1" and mn == "Línea base (prior)":
                continue
            oof = np.zeros(len(tr))
            cvm = []
            for itr, iva in folds:
                pp = _pipe(esc, clone(clf)).fit(Xtr.iloc[itr], ytr[itr])
                oof[iva] = pp.predict_proba(Xtr.iloc[iva])[:, 1]
                cvm.append((roc_auc_score(ytr[iva], oof[iva]), average_precision_score(ytr[iva], oof[iva])))
            umbral = _umbral_recall(ytr, oof) if mn != "Línea base (prior)" else 0.5
            final = _pipe(esc, clone(clf)).fit(Xtr, ytr)
            pte = final.predict_proba(Xte)[:, 1]
            m = _metricas(yte, pte, umbral)
            lo, hi = _boot_grupos(yte, pte, gte, np.random.default_rng(SEED)) if mn != "Línea base (prior)" else (0.5, 0.5)
            cvm = np.array(cvm)
            filas.append({"escenario": en, "descripcion": esc["desc"], "modelo": mn,
                          "cv_auc_roc": cvm[:, 0].mean(), "cv_auc_roc_de": cvm[:, 0].std(ddof=1),
                          "cv_auc_pr": cvm[:, 1].mean(), "cv_auc_pr_de": cvm[:, 1].std(ddof=1),
                          "umbral": umbral, **m, "auc_ic_lo": lo, "auc_ic_hi": hi})
            preds[(en, mn)] = pte
            ajustados[(en, mn)] = final
            log.info("Modelo %s | %s: AUC test %.3f", en, mn, m["auc_roc"])
    # Redondeo a 6 decimales: elimina ruido de coma flotante (ulps) de los cálculos
    # multihilo (OpenMP) y hace estable la verificación por hash; el informe usa 3.
    res = pd.DataFrame(filas).round(6)
    res.to_csv(RESULTS / "modelo_metricas.csv", index=False)

    # Sensibilidad: diferencia de AUC (A + sim) − A con bootstrap pareado por estudiante
    sens = []
    for mn in ["Regresión logística", "Random forest", "Gradient boosting"]:
        for alt, base in (("A_mas_sim", "A_corte_P1"), ("C_mas_sim", "C_matricula"),
                          ("A_mas_faltas", "A_corte_P1"), ("B_corte_P2", "A_corte_P1"),
                          ("A_corte_P1", "C_matricula")):
            p_alt, p_base = preds[(alt, mn)], preds[(base, mn)]
            d = roc_auc_score(yte, p_alt) - roc_auc_score(yte, p_base)
            lo, hi = _boot_grupos(yte, p_alt, gte, np.random.default_rng(SEED), p2=p_base)
            sens.append({"modelo": mn, "comparacion": f"{alt} − {base}", "delta_auc": d,
                         "ic_lo": lo, "ic_hi": hi})
    sens = pd.DataFrame(sens).round(6)
    sens.to_csv(RESULTS / "modelo_sensibilidad.csv", index=False)

    # Contrafactual didáctico: generador CIRCULAR (LMS que depende de G3) -> AUC inflada
    rng_c = np.random.default_rng(SEED)
    circ = df.copy()
    circ["sim_circular"] = np.exp(0.15 * circ.nota_p3 + rng_c.normal(0, 0.5, len(circ)))
    ctr, cte = circ[circ.particion == "train"], circ[circ.particion == "test"]
    circular = {}
    for nombre, base in (("C", ESCENARIOS["C_matricula"]), ("A", ESCENARIOS["A_corte_P1"])):
        esc_c = {"num": base["num"] + ["sim_circular"], "bin": base["bin"]}
        pc = _pipe(esc_c, LogisticRegression(max_iter=2000)).fit(
            _X(ctr, esc_c), ctr.en_riesgo).predict_proba(_X(cte, esc_c))[:, 1]
        e0 = "C_matricula" if nombre == "C" else "A_corte_P1"
        circular[f"auc_{nombre}_honesto"] = roc_auc_score(yte, preds[(e0, "Regresión logística")])
        circular[f"auc_{nombre}_mas_sim_honesto"] = roc_auc_score(yte, preds[(("C_mas_sim" if nombre == "C" else "A_mas_sim"), "Regresión logística")])
        circular[f"auc_{nombre}_mas_circular"] = roc_auc_score(cte.en_riesgo, pc)
    pd.DataFrame([circular]).to_csv(RESULTS / "modelo_contrafactual_circular.csv", index=False)

    # Resultados por escuela y asignatura (modelo principal: A, regresión logística)
    principal_m = ("A_corte_P1", "Regresión logística")
    um = res[(res.escenario == principal_m[0]) & (res.modelo == principal_m[1])].umbral.iloc[0]
    estr = []
    for col in ("escuela", "asignatura"):
        for v in te[col].unique():
            mk = (te[col] == v).values
            if len(np.unique(yte[mk])) < 2:
                continue
            mm = _metricas(yte[mk], preds[principal_m][mk], um)
            estr.append({"estrato": col, "valor": v, "n": int(mk.sum()), "prevalencia": yte[mk].mean(),
                         "auc_roc": mm["auc_roc"], "recall": mm["recall"], "precision": mm["precision"]})
    estr = pd.DataFrame(estr)
    estr.to_csv(RESULTS / "modelo_estratos.csv", index=False)

    # Interpretación 1: OR con statsmodels (conjunto parsimonioso preespecificado, train)
    cols_or = ["nota_p1", "reprobaciones_previas", "tiempo_estudio", "edad", "salidas", "alcohol_finde"]
    Xo = tr[cols_or].astype(float).copy()
    Xo["asignatura_POR"] = (tr.asignatura == "POR").astype(float)
    Xo["escuela_MS"] = (tr.escuela == "MS").astype(float)
    Xo["sexo_M"] = (tr.sexo == "M").astype(float)
    Xo["desea_superior"] = tr.desea_superior.astype(float)
    Xo["apoyo_escolar"] = tr.apoyo_escolar.astype(float)
    Xo["zona_rural"] = (tr.zona == "Rural").astype(float)
    glm = sm.GLM(ytr.astype(float), sm.add_constant(Xo), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": gtr})
    ci = glm.conf_int()
    ors = pd.DataFrame({"OR": np.exp(glm.params), "ic_lo": np.exp(ci[0]), "ic_hi": np.exp(ci[1]),
                        "p": glm.pvalues}).drop("const")
    ors.to_csv(RESULTS / "modelo_odds_ratios.csv")

    # Interpretación 2: importancia por permutación (test) del modelo no lineal en A
    mejor_nl = max(["Random forest", "Gradient boosting"],
                   key=lambda mn: res[(res.escenario == "A_corte_P1") & (res.modelo == mn)].cv_auc_roc.iloc[0])
    esc = ESCENARIOS["A_corte_P1"]
    pi = permutation_importance(ajustados[("A_corte_P1", mejor_nl)], _X(te, esc), yte, n_repeats=30,
                                random_state=SEED, scoring="roc_auc")
    imp = pd.DataFrame({"variable": _X(te, esc).columns, "importancia": pi.importances_mean,
                        "de": pi.importances_std}).sort_values("importancia", ascending=False)
    imp.to_csv(RESULTS / "modelo_importancia_permutacion.csv", index=False)

    # Figuras de modelo
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for mn, col in (("Regresión logística", AZUL), ("Random forest", NARANJA), ("Gradient boosting", AQUA)):
        p = preds[("A_corte_P1", mn)]
        fpr, tpr, _ = roc_curve(yte, p)
        axs[0].plot(fpr, tpr, color=col, lw=2, label=f"{mn} ({roc_auc_score(yte, p):.2f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        axs[1].plot(rc, pr, color=col, lw=2, label=f"{mn} ({average_precision_score(yte, p):.2f})")
    axs[0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1, label="Línea base (0.50)")
    axs[1].axhline(yte.mean(), color=GRIS, ls="--", lw=1, label=f"Línea base ({yte.mean():.2f})")
    axs[0].set(xlabel="1 − especificidad", ylabel="Sensibilidad (recall)", title="ROC (test, corte P1)")
    axs[1].set(xlabel="Recall", ylabel="Precisión", title="Precisión–recall (test, corte P1)")
    for ax in axs:
        ax.legend(fontsize=6.5, loc="lower right" if ax is axs[0] else "lower left")
        ax.title.set_color(TINTA)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_modelo_roc_pr.png")
    plt.close(fig)

    s = res[res.modelo == "Regresión logística"].set_index("escenario").loc[
        ["C_matricula", "C_mas_sim", "A_corte_P1", "A_mas_sim", "A_mas_faltas", "B_corte_P2"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(s))[::-1]
    cols = [NARANJA if i.endswith("_sim") else AZUL for i in s.index]
    ax.hlines(y, s.auc_ic_lo, s.auc_ic_hi, color=cols, lw=2)
    ax.scatter(s.auc_roc, y, color=cols, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    for yi, r in zip(y, s.itertuples()):
        ax.text(r.auc_ic_hi + 0.01, yi, f"{r.auc_roc:.3f}", va="center", fontsize=8, color=TINTA2)
    ax.set_yticks(y, [ESCENARIOS[i]["desc"] for i in s.index], fontsize=7.5)
    ax.set_xlabel("AUC-ROC en test (regresión logística), IC 95 % bootstrap por estudiante", fontsize=8)
    ax.set_xlim(0.5, 1.06)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_modelo_escenarios.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for mn, col in (("Regresión logística", AZUL), ("Random forest", NARANJA), ("Gradient boosting", AQUA)):
        fp, mp = calibration_curve(yte, preds[("A_corte_P1", mn)], n_bins=5, strategy="quantile")
        axs[0].plot(mp, fp, marker="o", ms=5, color=col, label=mn)
    axs[0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1)
    axs[0].set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Calibración (test, corte P1)")
    axs[0].legend(fontsize=7)
    top = imp.head(8)[::-1]
    axs[1].barh(top.variable, top.importancia, xerr=top.de, color=AZUL, height=0.55,
                error_kw={"ecolor": TINTA2, "lw": 1})
    axs[1].set(xlabel="Caída media de AUC al permutar", title=f"Importancia por permutación\n({mejor_nl}, test)")
    axs[1].tick_params(axis="y", labelsize=7.5)
    axs[1].grid(axis="y", visible=False)
    for ax in axs:
        ax.title.set_color(TINTA)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_modelo_calibracion_importancia.png")
    plt.close(fig)
    return {"metricas": res, "sensibilidad": sens, "odds": ors, "importancia": imp, "estratos": estr,
            "circular": circular, "mejor_nl": mejor_nl}


# --------------------------------------------------------------------------- #
# Texto libre (demostración, datos simulados)
# --------------------------------------------------------------------------- #
def texto() -> dict:
    from pymongo import MongoClient
    from config import MONGO_DB, MONGO_URI
    from transform_synthetic import tokens
    db = MongoClient(MONGO_URI)[MONGO_DB]
    verdad = json.loads((DATA_SYNTH_RAW / "_verdad_generador.json").read_text(encoding="utf-8"))["categoria_real_por_nota"]
    notas = list(db.alertas_riesgo.aggregate([{"$unwind": "$notas"}, {"$project": {
        "_id": 0, "nota_id": "$notas.nota_id", "texto": "$notas.texto",
        "norm": "$notas.texto_normalizado", "cat": "$notas.categoria"}}]))
    t = pd.DataFrame(notas)
    t["real"] = t.nota_id.map(verdad)
    acc = float((t.cat == t.real).mean())
    conf = pd.crosstab(t.real, t.cat)
    conf.to_csv(RESULTS / "texto_confusion.csv")
    from collections import Counter
    terms = Counter(w for n in t.norm for w in tokens(n))
    top = pd.DataFrame(terms.most_common(15), columns=["termino", "frecuencia"])
    top.to_csv(RESULTS / "texto_terminos.csv", index=False)
    ejemplos = t.sample(4, random_state=SEED)[["texto", "norm", "cat"]]
    ejemplos.to_csv(RESULTS / "texto_ejemplos.csv", index=False)
    return {"n_notas": len(t), "exactitud": acc, "indeterminadas": int((t.cat == "indeterminada").sum())}


def run() -> dict:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(DATA_PROCESSED / "dataset_analitico.parquet")
    desc = descriptivos(df)
    r1 = p1(df, rng)
    r2 = p2(df, rng)
    rp = principal(df)
    rm = modelado(df, rng)
    rt = texto()
    resumen = {"descriptivos": desc.to_dict("records"), "circular": rm["circular"], "texto": rt,
               "mejor_no_lineal": rm["mejor_nl"]}
    with open(RESULTS / "resumen_analisis.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=float)
    etapa("analysis", figuras=len(list(FIGURES.glob("fig_*.png"))), exactitud_texto=round(rt["exactitud"], 4),
          **{"auc_test_A_logistica": round(float(rm["metricas"].query(
              "escenario=='A_corte_P1' and modelo=='Regresión logística'").auc_roc.iloc[0]), 4)})
    return {"p1": r1, "p2": r2, "principal": rp, "modelo": rm, "texto": rt}


if __name__ == "__main__":
    run()
