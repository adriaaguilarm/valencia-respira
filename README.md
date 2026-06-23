# Valencia Respira

Aplicación Streamlit para consultar la calidad del aire de Valencia, generar predicciones a 8 horas y explorar el histórico por estación y contaminante.

## Ejecución local

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La aplicación necesita estas variables de entorno:

- `MISTRAL_API_KEY`
- `EDM_GITHUB_TOKEN`

## Streamlit Community Cloud

- Entrypoint: `app/streamlit_app.py`
- Python: `3.11`
- Dependencias Python: `requirements.txt`
- Dependencias del sistema: `packages.txt`


