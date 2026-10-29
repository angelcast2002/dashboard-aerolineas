import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ============================ CONFIGURACIÓN ============================
# URL: https://aerolineas-uvg-bg.streamlit.app/
DSN = "HiveQuickstart"
BASE_HIVE = "vuelos_db"
TABLAS = ["res_aerolinea_anio", "res_causas", "res_flota"]
CARPETA = Path(__file__).parent
CARPETA_CSV = CARPETA / "datos_hive"
CARPETA_SALIDA = CARPETA / "resultados_python"
VENTANA = (2004, 2007)

NARANJA, GRIS, GRIS_OSC, AZUL = "#E8713A", "#C4C4C4", "#5F6368", "#3B6EA8"
COLOR_CAUSA = {"Aerolínea": "#3B6EA8", "Avión tarde": "#1F3A5F", "Clima": "#F2C14E",
               "Sistema NAS": "#B279A2", "Seguridad": "#9E9E9E"}
COLOR_SEG = {"Alto volumen eficiente en expansión": NARANJA, "Nicho de alta calidad": "#54A24B",
             "Low cost jóvenes en crecimiento acelerado": AZUL,
             "Tradicionales grandes en contracción": GRIS_OSC, "Regionales con baja confiabilidad": "#B279A2"}
INDICADORES = {"participacion_2007": "Escala (participación 2007)", "cagr": "Crecimiento 2004-2007",
               "puntualidad": "Puntualidad", "cancelacion": "Baja cancelación",
               "retraso_prom": "Bajo retraso", "edad_flota": "Flota joven"}
PESOS_BASE = {"participacion_2007": 25, "cagr": 20, "puntualidad": 20,
              "cancelacion": 15, "retraso_prom": 10, "edad_flota": 10}
MENOR_ES_MEJOR = {"cancelacion", "retraso_prom", "edad_flota"}

st.set_page_config(page_title="Inversión en aerolíneas", page_icon="✈️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
.kpi {background: #FFFFFF; border: 1px solid #ECECEC; border-radius: 12px; padding: 14px 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,.05); height: 100%;}
.kpi .t {font-size: .78rem; color: #6B6B6B; text-transform: uppercase; letter-spacing: .04em;}
.kpi .v {font-size: 1.65rem; font-weight: 700; color: #1F1F1F; line-height: 1.25;}
.kpi .s {font-size: .78rem; color: #8A8A8A;}
.kpi.foco {border-top: 4px solid #E8713A;}
.badge {display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem;
        background:#F1F3F4; color:#3C4043; margin-right:6px;}
h1 {font-size: 1.9rem !important;}
</style>""", unsafe_allow_html=True)


def mostrar(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def estilo(fig, alto=420):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Segoe UI, Arial", size=12), title_font_size=15,
                      legend=dict(orientation="h", y=-0.15, x=0, title_text=""))
    fig.update_layout(margin=dict(r=45))
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    return fig


# ============================ DATOS ============================
def _leer_hive(tabla):
    import pyodbc
    con = pyodbc.connect(f"DSN={DSN}", autocommit=True, timeout=20)
    cur = con.cursor()
    cur.execute(f"SELECT * FROM {BASE_HIVE}.{tabla}")
    df = pd.DataFrame.from_records(cur.fetchall(), columns=[d[0] for d in cur.description])
    con.close()
    return df


@st.cache_data(show_spinner="Consultando Hive...")
def cargar(usar_hive):
    datos, origen, t0 = {}, "CSV local", time.time()
    for t in TABLAS:
        df = None
        if usar_hive:
            try:
                df = _leer_hive(t)
                origen = f"Hive · {BASE_HIVE} vía ODBC"
                CARPETA_CSV.mkdir(exist_ok=True)
            except Exception:
                df, origen = None, "CSV local (Hive no disponible)"
        if df is None:
            df = pd.read_csv(CARPETA_CSV / f"{t}.csv")
        df.columns = [str(c).split(".")[-1].strip().lower() for c in df.columns]
        if usar_hive and origen.startswith("Hive"):
            df.to_csv(CARPETA_CSV / f"{t}.csv", index=False)
        datos[t] = df
    v, c, f = datos["res_aerolinea_anio"], datos["res_causas"], datos["res_flota"]
    for d, excl in ((v, ("codigo", "aerolinea")), (c, ("codigo",)), (f, ("codigo",))):
        cols = [x for x in d.columns if x not in excl]
        d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    return v, c, f, origen, time.time() - t0


def nombre_corto(n):
    n = str(n).split(" (")[0]
    for s in (" Inc.", " Co.", " Corporation", " LLC"):
        n = n.replace(s, "")
    return n.strip()


@st.cache_data
def preparar(v, c, f):
    cat = v.groupby(["codigo", "aerolinea"]).agg(ini=("anio", "min"), fin=("anio", "max")).reset_index()
    cat["nombre"] = cat.aerolinea.map(nombre_corto)
    nombre = dict(zip(cat.codigo, cat.nombre))
    activas = set(cat.loc[cat.fin == 2008, "codigo"])
    piv = v.pivot_table(index="codigo", columns="anio", values="vuelos")

    a0, a1 = VENTANA
    w = v[v.anio.between(a0, a1)]
    n_anios = w.groupby("codigo").anio.nunique()
    universo = sorted(set(n_anios[n_anios == a1 - a0 + 1].index) & activas)
    g = w[w.codigo.isin(universo)].groupby("codigo").sum(numeric_only=True)

    ind = pd.DataFrame(index=g.index)
    ind["nombre"] = ind.index.map(nombre)
    ind["vuelos_2007"] = piv.loc[ind.index, a1]
    ind["participacion_2007"] = ind.vuelos_2007 / v.loc[v.anio == a1, "vuelos"].sum()
    ind["cagr"] = (piv.loc[ind.index, a1] / piv.loc[ind.index, a0]) ** (1 / (a1 - a0)) - 1
    ind["puntualidad"] = g.puntuales / g.con_llegada
    ind["cancelacion"] = g.cancelados / g.vuelos
    ind["retraso_prom"] = g.suma_retraso_llegada / g.con_llegada
    cc = c[c.anio.between(a0, a1) & c.codigo.isin(universo)].groupby("codigo").sum(numeric_only=True)
    tot = cc[["min_aerolinea", "min_avion_tarde", "min_clima", "min_sistema", "min_seguridad"]].sum(axis=1)
    ind["controlable"] = (cc.min_aerolinea + cc.min_avion_tarde) / tot
    ff = f.set_index("codigo")
    ind["aviones"] = ff.aviones.reindex(ind.index)
    ind["edad_flota"] = ff.edad_prom.reindex(ind.index)

    X = ind[list(INDICADORES)].fillna(ind[list(INDICADORES)].median())
    norm = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)
    for col in MENOR_ES_MEJOR:
        norm[col] = 1 - norm[col]

    # Segmentación K-means
    Z = ind[["vuelos_2007", "cagr", "puntualidad", "cancelacion", "retraso_prom", "edad_flota"]].copy()
    Z["vuelos_2007"] = np.log(Z.vuelos_2007)
    Zs = StandardScaler().fit_transform(Z.fillna(Z.median()))
    sil = {k: silhouette_score(Zs, KMeans(k, n_init=50, random_state=42).fit(Zs).labels_) for k in range(3, 7)}
    k = max(sil, key=sil.get)
    lab = KMeans(k, n_init=50, random_state=42).fit(Zs).labels_
    ind["cluster"] = lab
    nombres = {}
    for cod, et in [("WN", "Alto volumen eficiente en expansión"), ("HA", "Nicho de alta calidad"),
                    ("B6", "Low cost jóvenes en crecimiento acelerado"),
                    ("DL", "Tradicionales grandes en contracción"), ("MQ", "Regionales con baja confiabilidad")]:
        if cod in ind.index:
            nombres.setdefault(ind.loc[cod, "cluster"], et)
    ind["segmento"] = ind.cluster.map(lambda x: nombres.get(x, f"Grupo {x + 1}"))

    # Proyección lineal 2008-2010
    hist, proy = np.arange(a0, a1 + 1), np.arange(a1 + 1, a1 + 4)
    filas = []
    for cod in universo:
        y = piv.loc[cod, hist].values.astype(float)
        reg = LinearRegression().fit(hist.reshape(-1, 1), y)
        filas += [(cod, nombre[cod], a, val, "Real") for a, val in zip(hist, y)]
        filas += [(cod, nombre[cod], a, val, "Proyectado")
                  for a, val in zip(proy, reg.predict(proy.reshape(-1, 1)).clip(min=0))]
    proyeccion = pd.DataFrame(filas, columns=["codigo", "nombre", "anio", "vuelos", "tipo"])
    ind["vuelos_2010_proy"] = proyeccion[proyeccion.anio == a1 + 3].set_index("codigo").vuelos

    return cat, nombre, piv, ind, norm, proyeccion, sil, k


@st.cache_data
def montecarlo(norm, pesos, n=10000, conc=20):
    rng = np.random.default_rng(42)
    p = np.array([pesos[k] for k in norm.columns])
    W = rng.dirichlet(p * conc + 1e-3, size=n)
    r = (-(norm.values @ W.T)).argsort(axis=0).argsort(axis=0) + 1
    return pd.DataFrame({"primero": (r == 1).mean(axis=1), "top3": (r <= 3).mean(axis=1)}, index=norm.index)


# ============================ BARRA LATERAL ============================
with st.sidebar:
    st.markdown("### ✈️ Controles")
    usar_hive = st.toggle("Leer desde Hive", value=True)
    if st.button("Recargar datos"):
        st.cache_data.clear()

v, c, f, origen, segs = cargar(usar_hive)
cat, NOMBRE, piv, ind, norm, proyeccion, sil, K = preparar(v, c, f)

with st.sidebar:
    opciones = ind.sort_values("vuelos_2007", ascending=False).index.tolist()
    foco = st.selectbox("Aerolínea a evaluar", opciones, index=opciones.index("WN") if "WN" in opciones else 0,
                        format_func=lambda x: NOMBRE[x])
    rango = st.slider("Años", 1987, 2008, (1988, 2007))
    top_n = st.slider("Aerolíneas en tendencias", 3, 12, 6)
    st.markdown("### ⚖️ Pesos del score")
    pesos_raw = {k: st.slider(INDICADORES[k], 0, 50, PESOS_BASE[k], step=5) for k in INDICADORES}

total = sum(pesos_raw.values()) or 1
pesos = {k: x / total for k, x in pesos_raw.items()}
ind["score"] = (norm * pd.Series(pesos)).sum(axis=1) * 100
ind["ranking"] = ind.score.rank(ascending=False, method="min").astype(int)
mc = montecarlo(norm, pesos)
ind = ind.join(mc)
CF = lambda cod: NARANJA if cod == foco else GRIS
mapa_color = {NOMBRE[x]: CF(x) for x in ind.index}

# ============================ ENCABEZADO ============================
st.title("¿Qué aerolínea adquirir?")
st.markdown(f"<span class='badge'>Fuente: {origen}</span><span class='badge'>DataExpo 2009 · 1987-2008</span>"
            f"<span class='badge'>{int(v.vuelos.sum()):,} vuelos</span><span class='badge'>{v.codigo.nunique()} aerolíneas</span>",
            unsafe_allow_html=True)
st.write("")

r = ind.loc[foco]
kpis = [("Score de inversión", f"{r.score:.1f}", f"Ranking #{r.ranking} de {len(ind)}"),
        ("1er lugar en simulación", f"{r.primero:.0%}", f"Top 3: {r.top3:.0%}"),
        ("Participación 2007", f"{r.participacion_2007:.1%}", f"{r.vuelos_2007:,.0f} vuelos"),
        ("Crecimiento anual", f"{r.cagr:+.1%}", "2004-2007"),
        ("Puntualidad", f"{r.puntualidad:.1%}", f"Retraso prom. {r.retraso_prom:.1f} min"),
        ("Cancelación", f"{r.cancelacion:.2%}", "2004-2007"),
        ("Flota", f"{r.aviones:,.0f}", f"{r.edad_flota:.1f} años prom." if pd.notna(r.edad_flota) else "sin dato de edad")]
cols = st.columns(len(kpis))
for col, (t, val, s) in zip(cols, kpis):
    col.markdown(f"<div class='kpi foco'><div class='t'>{t}</div><div class='v'>{val}</div>"
                 f"<div class='s'>{s}</div></div>", unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["🏆 Score", "📈 Mercado", "⏱️ Operación", "🔍 Causas de retraso", "🛩️ Flota", "🧩 Segmentos", "🔮 Proyección"])

# ---------------- SCORE ----------------
with tabs[0]:
    a, b = st.columns([1.1, 1])
    o = ind.sort_values("score")
    fig = go.Figure(go.Bar(x=o.score, y=o.nombre, orientation="h", marker_color=[CF(x) for x in o.index],
                           text=o.score.round(1), textposition="outside"))
    fig.update_xaxes(range=[0, 100])
    with a:
        mostrar(estilo(fig.update_layout(title="Score de inversión (0-100)"), 560))
    top = ind.sort_values("ranking").index[:6]
    ap = (norm.loc[top] * pd.Series(pesos) * 100).rename(columns=INDICADORES)
    ap.index = ap.index.map(NOMBRE)
    ap = ap.reset_index().melt(id_vars="codigo", var_name="Indicador", value_name="Puntos")
    fig = px.bar(ap, x="Puntos", y="codigo", color="Indicador", orientation="h",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_yaxes(categoryorder="total ascending", title="")
    fig.update_xaxes(title="")
    with b:
        mostrar(estilo(fig.update_layout(title="Composición del score: top 6"), 320))
        m = ind[ind.top3 > 0].sort_values("top3")
        fig = go.Figure()
        fig.add_bar(x=m.top3 * 100, y=m.nombre, orientation="h", name="Top 3", marker_color="#E3E3E3")
        fig.add_bar(x=m.primero * 100, y=m.nombre, orientation="h", name="1er lugar",
                    marker_color=[CF(x) if x == foco else GRIS_OSC for x in m.index])
        fig.update_layout(barmode="overlay", xaxis=dict(range=[0, 100], ticksuffix="%"))
        mostrar(estilo(fig.update_layout(title="Monte Carlo: 10,000 combinaciones de pesos"), 270))

# ---------------- MERCADO ----------------
with tabs[1]:
    vr = v[v.anio.between(*rango)].copy()
    vr["nombre"] = vr.codigo.map(NOMBRE)
    grandes = vr.groupby("codigo").vuelos.sum().nlargest(top_n).index
    d = vr[vr.codigo.isin(grandes)].sort_values("anio")
    fig = px.line(d, x="anio", y="vuelos", color="nombre", color_discrete_map=mapa_color, markers=True)
    fig.for_each_trace(lambda t: t.update(line_width=4 if t.name == NOMBRE[foco] else 2))
    fig.update_xaxes(title="", dtick=2)
    fig.update_yaxes(title="Vuelos")
    mostrar(estilo(fig.update_layout(title=f"Vuelos por año · {top_n} aerolíneas más grandes"), 430))
    a, b = st.columns(2)
    ult = vr[vr.anio == min(rango[1], 2007)]
    cuota = (ult.groupby("codigo").vuelos.sum() / ult.vuelos.sum()).nlargest(12).sort_values()
    fig = go.Figure(go.Bar(x=cuota * 100, y=cuota.index.map(NOMBRE), orientation="h",
                           marker_color=[CF(x) for x in cuota.index], text=(cuota * 100).round(1),
                           texttemplate="%{text}%", textposition="outside"))
    with a:
        mostrar(estilo(fig.update_layout(title=f"Participación de mercado {min(rango[1], 2007)}",
                                         xaxis_ticksuffix="%"), 420))
    tot = v.groupby("anio").vuelos.sum().reset_index()
    tot["tipo"] = np.where(tot.anio.isin([1987, 2008]), "Año parcial", "Año completo")
    fig = px.bar(tot, x="anio", y="vuelos", color="tipo",
                 color_discrete_map={"Año parcial": NARANJA, "Año completo": "#9AA0A6"})
    with b:
        mostrar(estilo(fig.update_layout(title="Vuelos totales del mercado por año", xaxis_title="",
                                         yaxis_title="Vuelos"), 420))

# ---------------- OPERACIÓN ----------------
with tabs[2]:
    a, b = st.columns([1.3, 1])
    d = ind.reset_index()
    fig = px.scatter(d, x="puntualidad", y="cancelacion", size="vuelos_2007", text="nombre", size_max=55,
                     color="nombre", color_discrete_map=mapa_color)
    fig.update_traces(textposition="top center", textfont_size=10, marker_line_color="white", marker_line_width=1)
    fig.add_vline(x=d.puntualidad.mean(), line_dash="dot", line_color=GRIS_OSC)
    fig.add_hline(y=d.cancelacion.mean(), line_dash="dot", line_color=GRIS_OSC)
    fig.update_layout(showlegend=False, xaxis=dict(tickformat=".0%", title="Puntualidad"),
                      yaxis=dict(tickformat=".1%", title="Cancelación"))
    with a:
        mostrar(estilo(fig.update_layout(title="Puntualidad vs. cancelación · 2004-2007"), 540))
    o = ind.sort_values("retraso_prom", ascending=False)
    fig = go.Figure(go.Bar(x=o.retraso_prom, y=o.nombre, orientation="h", marker_color=[CF(x) for x in o.index],
                           text=o.retraso_prom.round(1), textposition="outside"))
    with b:
        mostrar(estilo(fig.update_layout(title="Retraso promedio de llegada (min)"), 540))

# ---------------- CAUSAS ----------------
with tabs[3]:
    cc = c[c.anio.between(max(rango[0], 2003), rango[1]) & c.codigo.isin(ind.index)].groupby("codigo").sum(numeric_only=True)
    mp = {"min_aerolinea": "Aerolínea", "min_avion_tarde": "Avión tarde", "min_clima": "Clima",
          "min_sistema": "Sistema NAS", "min_seguridad": "Seguridad"}
    pct = cc[list(mp)].div(cc[list(mp)].sum(axis=1), axis=0).rename(columns=mp)
    pct = pct.loc[ind.sort_values("vuelos_2007", ascending=False).index[:12]]
    pct.index = pct.index.map(NOMBRE)
    d = pct.reset_index().melt(id_vars="codigo", var_name="Causa", value_name="pct")
    fig = px.bar(d, x="pct", y="codigo", color="Causa", orientation="h", color_discrete_map=COLOR_CAUSA,
                 text=d.pct.map(lambda x: f"{x:.0%}" if x >= 0.06 else ""))
    fig.update_yaxes(categoryorder="array", categoryarray=list(pct.index[::-1]), title="")
    fig.update_xaxes(tickformat=".0%", title="")
    a, b = st.columns([1.6, 1])
    with a:
        mostrar(estilo(fig.update_layout(title="Minutos de retraso por causa"), 540))
    o = ind.sort_values("controlable")
    fig = go.Figure(go.Bar(x=o.controlable * 100, y=o.nombre, orientation="h",
                           marker_color=[CF(x) for x in o.index], text=(o.controlable * 100).round(0),
                           texttemplate="%{text}%", textposition="outside"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    with b:
        mostrar(estilo(fig.update_layout(title="Retraso controlable por la aerolínea"), 540))

# ---------------- FLOTA ----------------
with tabs[4]:
    o = ind.sort_values("aviones", ascending=False)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=o.nombre, y=o.aviones, name="Aviones", marker_color=[CF(x) for x in o.index])
    fig.add_scatter(x=o.nombre, y=o.edad_flota, name="Edad promedio (años)", mode="lines+markers",
                    line=dict(color=AZUL, width=3), secondary_y=True)
    fig.update_yaxes(title_text="Aviones", secondary_y=False)
    fig.update_yaxes(title_text="Edad promedio (años)", secondary_y=True, showgrid=False)
    mostrar(estilo(fig.update_layout(title="Tamaño y antigüedad de la flota"), 500))

# ---------------- SEGMENTOS ----------------
with tabs[5]:
    d = ind.reset_index()
    d["cagr_pct"], d["punt_pct"] = d.cagr * 100, d.puntualidad * 100
    fig = px.scatter(d, x="cagr_pct", y="punt_pct", size="vuelos_2007", color="segmento", text="nombre",
                     color_discrete_map=COLOR_SEG, size_max=55)
    fig.update_traces(textposition="top center", textfont_size=10, marker_line_color="white", marker_line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color=GRIS_OSC)
    fig.update_layout(xaxis_title="Crecimiento anual 2004-2007 (%)", yaxis_title="Puntualidad (%)")
    a, b = st.columns([2.2, 1])
    with a:
        mostrar(estilo(fig.update_layout(title=f"Segmentos K-means (k={K})"), 560))
    s = pd.DataFrame({"k": list(sil), "silueta": list(sil.values())})
    fig = go.Figure(go.Bar(x=s.k.astype(str), y=s.silueta, text=s.silueta.round(3), textposition="outside",
                           marker_color=[NARANJA if x == K else GRIS for x in s.k]))
    with b:
        mostrar(estilo(fig.update_layout(title="Coeficiente de silueta", xaxis_title="k",
                                         yaxis_range=[0, s.silueta.max() * 1.25]), 270))
        res = ind.groupby("segmento").agg(Aerolíneas=("nombre", "size"), Crec=("cagr", "mean"),
                                          Punt=("puntualidad", "mean")).sort_values("Crec", ascending=False)
        res["Crec"] = res["Crec"].map(lambda x: f"{x:+.1%}")
        res["Punt"] = res["Punt"].map(lambda x: f"{x:.1%}")
        st.dataframe(res, height=240)

# ---------------- PROYECCIÓN ----------------
with tabs[6]:
    grandes = ind.vuelos_2007.nlargest(top_n).index
    fig = go.Figure()
    for cod in grandes:
        d = proyeccion[proyeccion.codigo == cod]
        real = d[d.tipo == "Real"]
        pr = pd.concat([real.tail(1), d[d.tipo == "Proyectado"]])
        ancho = 4 if cod == foco else 2
        fig.add_scatter(x=real.anio, y=real.vuelos, mode="lines+markers", name=NOMBRE[cod],
                        line=dict(color=CF(cod), width=ancho), legendgroup=cod)
        fig.add_scatter(x=pr.anio, y=pr.vuelos, mode="lines", line=dict(color=CF(cod), width=ancho, dash="dash"),
                        showlegend=False, legendgroup=cod)
        fig.add_annotation(x=pr.anio.iloc[-1], y=pr.vuelos.iloc[-1], text=NOMBRE[cod], xanchor="left",
                           showarrow=False, xshift=6, font=dict(color=CF(cod) if cod == foco else GRIS_OSC))
    fig.add_vrect(x0=VENTANA[1] + 0.5, x1=VENTANA[1] + 3.2, fillcolor="#F2F2F2", line_width=0, layer="below")
    fig.update_xaxes(dtick=1, range=[VENTANA[0] - 0.2, VENTANA[1] + 4.2])
    fig.update_layout(showlegend=False, yaxis_title="Vuelos")
    mostrar(estilo(fig.update_layout(title="Vuelos reales 2004-2007 y proyección lineal 2008-2010"), 520))

# ============================ EXPORTAR PARA POWER BI ============================
CARPETA_SALIDA.mkdir(exist_ok=True)
ind.reset_index().rename(columns={"index": "codigo"}).to_csv(CARPETA_SALIDA / "python_score.csv",
                                                              index=False, encoding="utf-8-sig")
proyeccion.to_csv(CARPETA_SALIDA / "python_proyeccion.csv", index=False, encoding="utf-8-sig")