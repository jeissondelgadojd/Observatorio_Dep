# ============================================================
# OBSERVATORIO SOCIOECONÓMICO DEPARTAMENTAL - COLOMBIA
# Versión visual mejorada + Resumen Ejecutivo flexible
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import warnings

from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    SPATIAL_AVAILABLE = True
except Exception:
    SPATIAL_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Observatorio Socioeconómico Departamental",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
    .big-title {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #c53030, #dd6b20, #d69e2e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem !important;
        line-height: 1.25 !important;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1rem !important;
        color: #a0aec0 !important;
        margin-bottom: 1.8rem !important;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">Observatorio Socioeconómico Departamental</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">35 indicadores · Periodo 2010-2023 · Versión visual mejorada</p>', unsafe_allow_html=True)

# ============================================================
# NORMALIZACIÓN Y DATOS
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None: return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "caldas": "Caldas", "caqueta": "Caquetá", "casanare": "Casanare", "cauca": "Cauca",
    "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba", "cundinamarca": "Cundinamarca",
    "guainia": "Guainía", "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo", "quindio": "Quindío",
    "risaralda": "Risaralda", "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia", "santander": "Santander",
    "sucre": "Sucre", "tolima": "Tolima", "valle": "Valle del Cauca",
    "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada",
}
LISTA_DEPTOS = sorted(list(set(DEPARTAMENTOS_CANONICOS.values())))

@st.cache_data(show_spinner="Cargando panel de indicadores...")
def generar_panel(seed=42):
    np.random.seed(seed)
    anios = list(range(2010, 2024))
    pesos = {
        "Bogotá D.C.": 25.5, "Antioquia": 14.8, "Valle del Cauca": 9.7, "Santander": 6.5,
        "Cundinamarca": 6.1, "Atlántico": 4.5, "Bolívar": 3.6, "Meta": 2.9, "Boyacá": 2.6,
        "Tolima": 2.2, "Córdoba": 1.9, "Huila": 1.8, "Cesar": 1.7, "Norte de Santander": 1.7,
        "Risaralda": 1.6, "Caldas": 1.5, "Nariño": 1.5, "Magdalena": 1.4, "Cauca": 1.4,
        "Casanare": 1.3, "La Guajira": 1.0, "Quindío": 0.9, "Sucre": 0.8, "Arauca": 0.55,
        "Caquetá": 0.5, "Chocó": 0.45, "Putumayo": 0.4, "San Andrés y Providencia": 0.15,
        "Guaviare": 0.12, "Amazonas": 0.10, "Vichada": 0.09, "Guainía": 0.07, "Vaupés": 0.05
    }
    for d in LISTA_DEPTOS:
        if d not in pesos: pesos[d] = 0.3

    n = len(LISTA_DEPTOS)
    f_econ = np.random.normal(0, 0.55, n)
    f_social = np.random.normal(0, 1.0, n)
    f_seg = -0.5 * f_econ + np.random.normal(0, 0.7, n)

    filas = []
    for i, dep in enumerate(LISTA_DEPTOS):
        w = pesos[dep]
        base_pib = w * 18000 * np.exp(0.12 * f_econ[i])
        for t, anio in enumerate(anios):
            tend = 0.021 * t
            ruido = np.random.normal(0, 0.045)
            filas.append({
                "departamento": dep, "anio": anio,
                "Tasa_Desempleo": np.clip(9.5 + 2.7*f_social[i] - 0.7*f_econ[i] - 0.15*t + np.random.normal(0,0.5), 3.5, 25),
                "Tasa_Ocupacion": np.clip(57 - 0.6*(9.5+2.7*f_social[i]) + 0.11*t + np.random.normal(0,1.1), 42, 72),
                "Informalidad": np.clip(47 + 7*f_social[i] - 2.8*f_econ[i] - 0.38*t + np.random.normal(0,1.4), 22, 74),
                "Brecha_Genero_Desempleo": np.clip(4.4 + 1.7*f_social[i] - 0.05*t + np.random.normal(0,0.4), 0.4, 12),
                "Tasa_Participacion": np.clip(61 + 3.2*f_econ[i] + 0.12*t + np.random.normal(0,1.2), 52, 73),
                "Subempleo": np.clip(10.5 + 3.8*f_social[i] - 0.16*t + np.random.normal(0,1.0), 5, 23),
                "Empleo_Formal": np.clip(53 - 6*f_social[i] + 2.5*f_econ[i] + 0.35*t + np.random.normal(0,1.3), 25, 78),
                "Desempleo_Juvenil": np.clip(15.5 + 4*f_social[i] - 0.17*t + np.random.normal(0,1.3), 8, 36),
                "PIB": base_pib * np.exp(tend + ruido),
                "PIB_per_capita": (base_pib * np.exp(tend + ruido)) / (w*170 + 45),
                "Indice_Competitividad": np.clip(46 + 15*f_econ[i] - 2.8*f_social[i] + 0.6*t + np.random.normal(0,1.9), 22, 92),
                "Exportaciones": (w*1100*np.exp(0.28*f_econ[i])) * np.exp(0.024*t + np.random.normal(0,0.08)),
                "Recaudo_Fiscal": (w*850*np.exp(0.22*f_econ[i])) * np.exp(0.026*t + np.random.normal(0,0.07)),
                "Inversion_Publica": (w*650*np.exp(0.25*f_econ[i])) * np.exp(0.023*t + np.random.normal(0,0.08)),
                "Indice_Innovacion": np.clip(33 + 18*f_econ[i] + 0.8*t + np.random.normal(0,2.1), 12, 88),
                "Productividad_Laboral": np.clip(39 + 16*f_econ[i] + 0.7*t + np.random.normal(0,2.0), 18, 86),
                "Comercio_Minorista": (w*800*np.exp(0.22*f_econ[i])) * np.exp(0.020*t + np.random.normal(0,0.08)),
                "Homicidios": max(3.2, np.exp(3.1 + 0.85*f_seg[i]) * np.exp(-0.028*t + np.random.normal(0,0.08))),
                "Violencia_Intrafamiliar": max(32, (155 + 42*f_seg[i]) * np.exp(-0.015*t + np.random.normal(0,0.06))),
                "IPM": np.clip(25.5 + 10.5*f_social[i] - 3.5*f_econ[i] - 0.55*t + np.random.normal(0,1.4), 6, 52),
                "Pobreza_Monetaria": np.clip(29 + 9.5*f_social[i] - 2.8*f_econ[i] - 0.5*t + np.random.normal(0,1.3), 9, 62),
                "Gini": np.clip(0.465 + 0.032*f_social[i] - 0.0016*t + np.random.normal(0,0.005), 0.39, 0.60),
                "Indice_Seguridad": np.clip(53 - 10*f_seg[i] + 5*f_econ[i] + 0.5*t + np.random.normal(0,2.5), 25, 91),
                "Desplazamiento": max(35, 650 + 260*f_seg[i] - 16*t + np.random.normal(0,60)),
                "Cobertura_Educacion": np.clip(68 + 11*f_econ[i] - 2.5*f_social[i] + 0.5*t + np.random.normal(0,1.6), 42, 97),
                "Anos_Escolaridad": np.clip(7.7 + 1.8*f_econ[i] + 0.06*t + np.random.normal(0,0.22), 5.5, 11.5),
                "Analfabetismo": np.clip(7.2 + 4.2*f_social[i] - 1.8*f_econ[i] - 0.2*t + np.random.normal(0,0.6), 1.5, 16),
                "Cobertura_Salud": np.clip(74 + 10*f_econ[i] - 2*f_social[i] + 0.42*t + np.random.normal(0,1.5), 48, 98),
                "Mortalidad_Infantil": np.clip(16.5 + 5*f_social[i] - 2.2*f_econ[i] - 0.35*t + np.random.normal(0,1.2), 6, 30),
                "Cobertura_Servicios": np.clip(71 + 13*f_econ[i] - 3.5*f_social[i] + 0.5*t + np.random.normal(0,1.5), 35, 97),
                "Penetracion_Internet": np.clip(41 + 17*f_econ[i] + 1.6*t + np.random.normal(0,2.0), 15, 94),
                "Cobertura_Agua": np.clip(77 + 12*f_econ[i] + 0.45*t + np.random.normal(0,1.6), 45, 98),
                "Cobertura_Energia": np.clip(85 + 10*f_econ[i] + 0.35*t + np.random.normal(0,1.3), 58, 99.5),
                "Indice_Vial": np.clip(43 + 15*f_econ[i] + 0.6*t + np.random.normal(0,2.5), 20, 90),
                "Turismo_Internacional": (w*170*np.exp(0.35*f_econ[i] - 0.25*f_seg[i])) * np.exp(0.027*t + np.random.normal(0,0.09)),
            })
    return pd.DataFrame(filas)

DF = generar_panel()
INDICADORES = [c for c in DF.columns if c not in ["departamento", "anio"]]

def filtrar(df, deptos, amin, amax):
    d = df[(df["anio"] >= amin) & (df["anio"] <= amax)].copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    return d

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🇨🇴 Filtros globales")
    deptos = st.multiselect("Departamentos", ["Todos"] + LISTA_DEPTOS, default=["Todos"])
    amin, amax = st.slider("Rango de años", 2010, 2023, (2010, 2023))
    st.markdown("---")

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🏠 Bienvenida y Contexto",
    "📊 Resumen Ejecutivo", 
    "🔍 Exploración", 
    "📈 EDA + PCA", 
    "📉 Econometría", 
    "🗺️ Espacial", 
    "🤖 Machine Learning", 
    "🎯 Simulador"
])

# ==================== TAB 0: BIENVENIDA Y CONTEXTO ====================
with tabs[0]:
    st.markdown("### Bienvenida y Contextualización del Observatorio")
    
    col_bienvenida1, col_bienvenida2 = st.columns([2, 1])
    
    with col_bienvenida1:
        st.markdown("""
        El **Observatorio Socioeconómico Departamental (Colombia)** es una plataforma de analítica avanzada e investigación cuantitativa diseñada para monitorear, diagnosticar y simular la dinámica territorial del país a nivel departamental durante el periodo 2010-2023.

        #### 🎯 Objetivos de la Plataforma
        - **Monitoreo Multidimensional:** Evaluar en tiempo real la evolución de 35 indicadores agrupados en dimensiones económicas, laborales, sociales, de seguridad e infraestructura.
        - **Análisis Avanzado:** Integrar técnicas econométricas clásicas (series de tiempo VAR) con métodos modernos de reducción de dimensionalidad (PCA) y aprendizaje automático (Ridge, LASSO, Random Forest, SVR, MLP).
        - **Evaluación de Impacto y Prospectiva:** Proporcionar herramientas interactivas de simulación contrafactual para el diseño y análisis de escenarios de política pública territorial.

        #### 🧩 Estructura Modular de la Plataforma
        1. **📊 Resumen Ejecutivo:** Visión panorámica y personalizada de los KPIs principales.
        2. **🔍 Exploración:** Filtrado, análisis de tendencias por departamento y descarga de series.
        3. **📈 EDA + PCA:** Matriz de correlaciones inter-indicador y reducción de dimensionalidad.
        4. **📉 Econometría:** Estimación de modelos Vectores Autorregresivos (VAR) y respuestas al impulso.
        5. **🗺️ Espacial:** Análisis de rankings territoriales e indicadores de autocorrelación espacial ($I$ de Moran).
        6. **🤖 Machine Learning:** Modelos predictivos supervisados y comparación de desempeño (RMSE, MAE, $R^2$).
        7. **🎯 Simulador:** Evaluación contrafactual de impacto ante variaciones en variables de política.
        """)

    with col_bienvenida2:
        st.info("""
        **Ficha Técnica del Sistema**
        - **Cobertura Territorial:** 32 Departamentos + Bogotá D.C.
        - **Dimensiones Analíticas:** Mercado Laboral, Macroeconomía, Competitividad, Seguridad, Pobreza, Educación y Servicios.
        - **Periodo de Análisis:** 2010 – 2023.
        - **Total Indicadores:** 35 variables departamentales.
        """)
        st.metric("Total Departamentos", len(LISTA_DEPTOS))
        st.metric("Indicadores Multidimensionales", len(INDICADORES))
        st.metric("Registros Históricos", len(DF))

# ==================== TAB 1: RESUMEN EJECUTIVO (CORREGIDO) ====================
with tabs[1]:
    st.markdown("### Resumen Ejecutivo configurable")

    kpis_sel = st.multiselect(
        "Selecciona los indicadores que quieres ver como KPI",
        INDICADORES,
        default=["PIB", "Tasa_Desempleo", "IPM", "Homicidios", "Informalidad", "Indice_Competitividad"],
        key="kpis_resumen"
    )

    d = filtrar(DF, deptos, amin, amax)

    if d.empty:
        st.warning("No hay datos con los filtros actuales. Ajusta el rango de años o los departamentos.")
    else:
        anio_kpi = int(d["anio"].max())
        ultimo = d[d["anio"] == anio_kpi]

        st.caption(f"Mostrando valores promedio del año **{anio_kpi}**")

        if kpis_sel and not ultimo.empty:
            n = len(kpis_sel)
            for i in range(0, n, 4):
                cols = st.columns(min(4, n - i))
                for j, ind in enumerate(kpis_sel[i:i+4]):
                    with cols[j]:
                        try:
                            val = float(ultimo[ind].mean())
                            label = ind.replace("_", " ")
                            if any(x in ind for x in ["PIB", "Exportaciones", "Recaudo", "Inversion", "Comercio", "Turismo"]):
                                st.metric(label, f"{val:,.0f}")
                            elif "Gini" in ind:
                                st.metric(label, f"{val:.3f}")
                            else:
                                st.metric(label, f"{val:.1f}")
                        except Exception:
                            st.metric(ind.replace("_", " "), "N/D")

        st.markdown("---")

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.markdown("#### Ranking departamental")
            ind_rank = st.selectbox("Indicador para ranking", INDICADORES, 
                                    index=INDICADORES.index("PIB") if "PIB" in INDICADORES else 0, 
                                    key="rank_sel")
            if not ultimo.empty:
                fig = px.bar(
                    ultimo.sort_values(ind_rank, ascending=True).tail(16),
                    x=ind_rank, y="departamento", orientation="h",
                    color=ind_rank, color_continuous_scale="Teal",
                    title=f"{ind_rank} ({anio_kpi})"
                )
                fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"}, 
                                  margin=dict(l=0,r=10,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### Evolución temporal")
            inds_evol = st.multiselect(
                "Indicadores a graficar", 
                INDICADORES, 
                default=["Tasa_Desempleo", "IPM", "Informalidad"],
                key="evol_sel"
            )
            if inds_evol:
                prom = d.groupby("anio")[inds_evol].mean().reset_index()
                fig = px.line(prom, x="anio", y=inds_evol, markers=True, title="Promedio de la selección")
                fig.update_layout(height=500, margin=dict(l=0,r=10,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver tabla resumen del año más reciente"):
            if not ultimo.empty and kpis_sel:
                st.dataframe(
                    ultimo.groupby("departamento")[kpis_sel].mean().round(2).sort_values(kpis_sel[0], ascending=False),
                    use_container_width=True
                )

# ==================== TAB 2: EXPLORACIÓN ====================
with tabs[2]:
    st.markdown("### Exploración detallada")
    ind = st.selectbox("Indicador", INDICADORES, key="explora_ind")
    d = filtrar(DF, deptos, amin, amax)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(d, x="anio", y=ind, color="departamento", markers=True, title=f"Tendencia – {ind}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ult = d[d["anio"] == amax].sort_values(ind, ascending=True) if not d.empty else pd.DataFrame()
        if not ult.empty:
            fig = px.bar(ult, x=ind, y="departamento", orientation="h", color=ind,
                         color_continuous_scale="YlOrRd", title=f"Ranking {amax}")
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    if not d.empty:
        csv = d[["departamento", "anio", ind]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar datos del indicador", csv, f"{ind}.csv", "text/csv")

# ==================== TAB 3: EDA + PCA ====================
with tabs[3]:
    st.markdown("### Análisis Exploratorio y Reducción de Dimensionalidad")
    d = filtrar(DF, deptos, amin, amax)
    n_comp = st.slider("Componentes PCA", 2, 8, 4, key="pca_slider")

    if len(d) > 30:
        corr = d[INDICADORES].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Matriz de correlaciones entre los 35 indicadores")
        st.plotly_chart(fig, use_container_width=True)

        X = StandardScaler().fit_transform(d[INDICADORES].dropna())
        pca = PCA(n_components=min(n_comp, X.shape[1])).fit(X)
        var = pca.explained_variance_ratio_

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var))], y=var, name="Individual"))
            fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(var))], y=np.cumsum(var),
                                     mode="lines+markers", name="Acumulada"))
            fig.update_layout(title="Scree Plot", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Varianza acumulada", f"{np.cumsum(var)[-1]*100:.1f}%")
            st.write("Los primeros componentes suelen capturar el eje desarrollo económico vs. vulnerabilidades sociales.")

# ==================== TAB 4: VAR ====================
with tabs[4]:
    st.markdown("### Econometría de Series de Tiempo (VAR)")
    vars_sel = st.multiselect("Variables para el VAR", INDICADORES,
                              default=["PIB", "Tasa_Desempleo", "IPM", "Homicidios"])
    if st.button("Estimar modelo VAR", key="btn_var") and len(vars_sel) >= 2:
        d = filtrar(DF, deptos, amin, amax)
        ts = d.groupby("anio")[vars_sel].mean().dropna()
        if len(ts) >= 12:
            res = VAR(ts).fit(maxlags=2, ic="aic")
            st.code(str(res.summary())[:1800])
            irf = res.irf(6)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=irf.irfs[:,0,1], mode="lines+markers", name="Respuesta"))
            fig.update_layout(title="Función de Impulso-Respuesta (ejemplo)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes observaciones temporales.")

# ==================== TAB 5: ESPACIAL ====================
with tabs[5]:
    st.markdown("### Análisis Espacial y Rankings")
    ind_e = st.selectbox("Indicador", INDICADORES, index=INDICADORES.index("PIB"), key="esp_ind")
    anio_e = st.slider("Año", 2010, 2023, 2022, key="esp_anio")
    d = DF[DF["anio"] == anio_e]

    fig = px.bar(d.sort_values(ind_e, ascending=True), x=ind_e, y="departamento", orientation="h",
                 color=ind_e, color_continuous_scale="YlOrRd", title=f"{ind_e} · {anio_e}")
    fig.update_layout(height=620, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    if SPATIAL_AVAILABLE and st.button("Calcular I de Moran"):
        dd = d.dropna(subset=[ind_e]).sort_values("departamento").reset_index(drop=True)
        w = KNN.from_dataframe(dd.assign(x=range(len(dd)), y=0), k=5)
        w.transform = "r"
        mi = Moran(dd[ind_e].values, w)
        st.success(f"**I de Moran = {mi.I:.4f}**  |  p-valor simulado = {mi.p_sim:.4f}")

# ==================== TAB 6: MACHINE LEARNING ====================
with tabs[6]:
    st.markdown("### Modelos de Machine Learning")
    target = st.selectbox("Variable objetivo", INDICADORES, index=INDICADORES.index("Tasa_Desempleo"))
    features = st.multiselect("Variables explicativas", [c for c in INDICADORES if c != target],
                              default=["PIB", "Informalidad", "IPM", "Indice_Competitividad", "Exportaciones"])
    modelo = st.selectbox("Algoritmo", ["Ridge", "LASSO", "Random Forest", "SVR", "MLP"])

    if st.button("Entrenar modelo") and features:
        d = filtrar(DF, deptos, amin, amax).dropna()
        X, y = d[features], d[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        if modelo == "Ridge":
            pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3,3,30)))])
        elif modelo == "LASSO":
            pipe = Pipeline([("sc", StandardScaler()), ("m", LassoCV(cv=5))])
        elif modelo == "Random Forest":
            pipe = Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1))])
        elif modelo == "SVR":
            pipe = Pipeline([("sc", StandardScaler()), ("m", SVR(C=10))])
        else:
            pipe = Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))])

        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{np.sqrt(mean_squared_error(yte, pred)):.3f}")
        m2.metric("MAE", f"{mean_absolute_error(yte, pred):.3f}")
        m3.metric("R²", f"{r2_score(yte, pred):.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yte, y=pred, mode="markers", opacity=0.65, name="Predicciones"))
        lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash"), name="Línea ideal"))
        fig.update_layout(title=f"{modelo} · Real vs Predicho", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 7: SIMULADOR ====================
with tabs[7]:
    st.markdown("### Simulador de Política Pública")
    target_s = st.selectbox("¿Qué quieres impactar?", INDICADORES, index=INDICADORES.index("Tasa_Desempleo"), key="sim_target")
    feats_s = st.multiselect("Variables de política que puedes modificar", INDICADORES,
                             default=["PIB", "Informalidad", "IPM", "Indice_Competitividad"], key="sim_feats")

    if feats_s:
        st.markdown("#### Define el escenario")
        cols = st.columns(min(4, len(feats_s)))
        valores = {}
        for i, f in enumerate(feats_s):
            with cols[i % len(cols)]:
                valores[f] = st.number_input(f.replace("_", " "), value=float(DF[f].mean()), format="%.2f", key=f"v_{f}")

        if st.button("Correr simulación", type="primary"):
            d = filtrar(DF, deptos, amin, amax).dropna()
            X, y = d[feats_s], d[target_s]
            pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
            pipe.fit(X, y)
            base_pred = pipe.predict(X.mean().values.reshape(1, -1))[0]
            esc_pred = pipe.predict(np.array([[valores[f] for f in feats_s]]))[0]
            delta = esc_pred - base_pred
            st.metric(label=f"Resultado simulado de {target_s}", value=f"{esc_pred:.2f}", delta=f"{delta:+.2f}")

st.markdown("---")
st.caption("Observatorio Socioeconómico Departamental · Colombia")
