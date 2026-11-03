# 🇨🇴 Observatorio Socioeconómico Departamental – Colombia

Aplicación interactiva en **Streamlit** para el análisis socioeconómico a nivel departamental en Colombia.

## Características principales

- **35 indicadores** multidimensionales (laborales, macroeconómicos, sociales, seguridad, educación, salud e infraestructura)
- Periodo: **2010 – 2023**
- Dashboard de resumen ejecutivo con KPIs
- Análisis Exploratorio + PCA
- Econometría de series de tiempo (VAR + Granger + IRF)
- Análisis espacial (Ranking + I de Moran)
- Machine Learning (Ridge, LASSO, Random Forest, SVR, MLP)
- Simulador de política pública + interpretabilidad SHAP
- Diseño moderno y responsive

## Instalación local

```bash
python -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub
2. Ve a [https://share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona el repositorio → Main file path: `app.py`
5. Deploy

## Estructura del proyecto

```
observatorio-departamental/
├── app.py
├── requirements.txt
├── README.md
└── data/                   # Para futuros archivos Excel del DANE
```

## Notas sobre los datos

El panel actual es **sintético calibrado** (con pesos económicos reales aproximados) porque la mayoría de series departamentales oficiales no están disponibles de forma limpia vía API pública.

Indicadores que sí se pueden obtener automáticamente en el futuro:
- PIB departamental (DANE)
- Homicidios, lesiones, hurtos (MinDefensa)
- Recursos de inversión (DNP)

Para pasar a datos 100% oficiales se recomienda incorporar los anexos Excel del DANE (GEIH, IPM, pobreza monetaria, cuentas departamentales).

## Autor

Desarrollado para seminario de investigación – Observatorio Socioeconómico Departamental (Colombia).
