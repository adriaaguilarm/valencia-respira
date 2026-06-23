from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from mistralai.client import Mistral


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".confing"
CURRENT_PATH = ROOT / "data" / "scraped" / "latest.csv"
PREDICTIONS_PATH = ROOT / "predictions" / "latest.csv"
MODEL_NAME = "mistral-large-latest"

POLLUTANTS = ["SO2", "NO2", "O3", "PM-10", "PM-2.5"]
QUALITY_LEVELS = [
    "Buena",
    "Razonablemente buena",
    "Regular",
    "Desfavorable",
    "Muy desfavorable",
    "Extremadamente desfavorable",
]
QUALITY_THRESHOLDS = {
    "SO2": [100, 200, 350, 500, 750, 1250],
    "NO2": [40, 90, 120, 230, 340, 1000],
    "O3": [50, 100, 130, 240, 380, 800],
    "PM-10": [20, 40, 50, 100, 150, 1200],
    "PM-2.5": [10, 20, 25, 50, 75, 800],
}

STATIC_CONTEXT = """
Eres un asistente para una interfaz de calidad del aire en Valencia. Responde siempre en espanol, de forma clara, breve y prudente.

Usa solo la informacion incluida en el contexto. Si no hay datos suficientes para una zona o contaminante, dilo explicitamente.
Distingue entre valores actuales scrapeados y valores predichos. No presentes predicciones como mediciones reales.

Los colores y categorias de calidad del aire se basan en el Indice Nacional de Calidad del Aire
(Orden TEC/351/2019, de 18 de marzo) y la Resolucion de 2 de septiembre de 2020.
Informacion de elaboracion propia basada en datos proporcionados por el Servicio de mejora climatica del Ayuntamiento de Valencia.

Umbrales por contaminante en ug/m3:
- SO2: Buena 0-100; Razonablemente buena 101-200; Regular 201-350; Desfavorable 351-500; Muy desfavorable 501-750; Extremadamente desfavorable 751-1250.
- NO2: Buena 0-40; Razonablemente buena 41-90; Regular 91-120; Desfavorable 121-230; Muy desfavorable 231-340; Extremadamente desfavorable 341-1000.
- O3: Buena 0-50; Razonablemente buena 51-100; Regular 101-130; Desfavorable 131-240; Muy desfavorable 241-380; Extremadamente desfavorable 381-800.
- PM-10: Buena 0-20; Razonablemente buena 21-40; Regular 41-50; Desfavorable 51-100; Muy desfavorable 101-150; Extremadamente desfavorable 151-1200.
- PM-2.5: Buena 0-10; Razonablemente buena 11-20; Regular 21-25; Desfavorable 26-50; Muy desfavorable 51-75; Extremadamente desfavorable 76-800.

Abreviaturas:
- ug/m3: microgramos por metro cubico.
- mg/m3: miligramos por metro cubico.
- PM-2.5: particulas en suspension inferiores a 2.5 micras.
- PM-10: particulas en suspension inferiores a 10 micras.
- NO2: Dioxido de Nitrogeno.
- SO2: Dioxido de Azufre.
- O3: Ozono.
""".strip()


def read_config_value(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"No se ha encontrado {name}. Define la variable de entorno o .confing.")
    match = re.search(rf"{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]", CONFIG_PATH.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"No se ha encontrado {name} en .confing.")
    return match.group(1)


def latest_valid_history_file(history_dir: Path) -> Path | None:
    for path in sorted(history_dir.glob("*.csv"), reverse=True):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not frame.empty:
            return path
    return None


def load_air_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        fallback = latest_valid_history_file(path.parent / "history")
        if fallback is not None:
            df = pd.read_csv(fallback, encoding="utf-8-sig")
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def quality_for_value(pollutant: str, value) -> str:
    if pd.isna(value):
        return "No hay datos"
    for level, upper in zip(QUALITY_LEVELS, QUALITY_THRESHOLDS[pollutant]):
        if float(value) <= upper:
            return level
    return "Extremadamente desfavorable"


def format_value(pollutant: str, value) -> str:
    if pd.isna(value):
        return "ND"
    return f"{float(value):.1f} ug/m3 ({quality_for_value(pollutant, value)})"


def format_markdown_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"### {title}\nSin datos disponibles."

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Zona": row["µg/m3"],
                **{pollutant: format_value(pollutant, row[pollutant]) for pollutant in POLLUTANTS},
            }
        )

    table = pd.DataFrame(rows)
    headers = list(table.columns)
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in table[col].fillna("ND")))
        for col in headers
    }
    header_line = "| " + " | ".join(str(col).ljust(widths[col]) for col in headers) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in headers) + " |"
    body = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in headers) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([f"### {title}", header_line, separator, *body])


def build_dynamic_context() -> str:
    current = load_air_table(CURRENT_PATH)
    predictions = load_air_table(PREDICTIONS_PATH)
    return "\n\n".join(
        [
            format_markdown_table(current, "TABLA ACTUAL SCRAPEADA"),
            format_markdown_table(predictions, "TABLA DE PREDICCIONES"),
        ]
    )


def build_user_prompt(question: str) -> str:
    return f"""
{STATIC_CONTEXT}

CONTEXTO DINAMICO DE DATOS
{build_dynamic_context()}

PREGUNTA DEL USUARIO
{question}

INSTRUCCIONES DE RESPUESTA
- Responde exclusivamente a lo que pregunta el usuario y ajusta el detalle al alcance de su consulta. No uses una plantilla fija.
- Si la pregunta pide un dato concreto de una zona o contaminante, responde en 1 a 3 frases, sin titulos ni listas innecesarias.
- Si pide una comparacion, un ranking o una vision general, organiza la informacion con una lista breve o titulos solo cuando mejoren la lectura.
- No anadas una seccion `Datos relevantes` por rutina. Usala unicamente si la consulta es amplia y hay varios hallazgos necesarios para responderla.
- Incluye predicciones solo si el usuario pregunta por el futuro, la evolucion o las proximas 8 horas. No anadas predicciones a preguntas sobre la situacion actual.
- Cuando uses una prediccion, identificala en la misma frase o seccion y aclara una sola vez que es una estimacion, no una medicion real.
- Menciona la falta de datos solo cuando afecte directamente a la zona, contaminante o comparacion preguntada; no enumeres ausencias ajenas a la consulta.
- Comprueba la coherencia antes de responder: nunca afirmes que falta un dato para una zona y contaminante si la tabla muestra un valor numerico para esa misma combinacion.
- Si no existe el dato exacto solicitado, dilo claramente y no lo sustituyas por otro contaminante, otra zona o una prediccion no pedida.
- Si comparas zonas, menciona los contaminantes concretos y sus categorias de calidad.
- No des consejo medico; limita la respuesta a interpretacion informativa de calidad del aire.
- Escribe en Markdown limpio y preparado para mostrarse directamente en la interfaz.
- Empieza por la respuesta, sin introducciones genericas, recapitulaciones ni frases de cierre innecesarias.
- Usa negrita y codigo inline con moderacion; reserva el codigo inline para valores numericos como `42.0 µg/m³`.
- No uses tablas ni encabezados de nivel 1 o 2.
""".strip()


def build_general_summary_prompt() -> str:
    return f"""
{STATIC_CONTEXT}

CONTEXTO DINAMICO DE DATOS
{build_dynamic_context()}

TAREA
Redacta un resumen general de la situacion de calidad del aire en Valencia para mostrar al inicio de la app.

INSTRUCCIONES DE RESPUESTA
- Devuelve exclusivamente Markdown siguiendo exactamente esta estructura:

### Estado general
**[categoria global].** [Una frase breve que resuma la situacion actual.]

### Puntos clave
- **[contaminante · zona]:** `[valor] µg/m³` — [categoria o interpretacion breve].
- Incluye entre 2 y 3 puntos, solo con los datos mas importantes.

### Prediccion +8 h
[Una frase breve sobre la evolucion prevista. Deja claro que es una prediccion y no una medicion.]

> **Cobertura:** [Menciona datos ausentes solo si afectan de forma relevante a la interpretacion.]

- La categoria global debe ser una de estas: buena, razonablemente buena, regular, desfavorable, muy desfavorable o extremadamente desfavorable.
- Si no hay ausencias relevantes, omite por completo la linea de cobertura.
- No uses tablas, parrafos largos, encabezados adicionales, introducciones ni conclusiones.
- No repitas el mismo dato en varias secciones.
- No des consejo medico; limita la respuesta a interpretacion informativa de calidad del aire.
""".strip()


def call_mistral_api(prompt: str, role: str, *, temperature: float = 0.2, max_tokens: int = 450) -> str:
    client = Mistral(api_key=read_config_value("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def ask_air_quality(question: str) -> str:
    role = (
        "Eres un asistente experto en explicar datos de calidad del aire de Valencia para usuarios generales. "
        "Responde al alcance exacto de cada pregunta, sin repetir una estructura fija ni anadir informacion no solicitada."
    )
    return call_mistral_api(build_user_prompt(question), role, temperature=0.2, max_tokens=500)


def summarize_air_quality() -> str:
    role = "Eres un asistente experto en resumir la situacion general de calidad del aire de Valencia para una interfaz publica."
    return call_mistral_api(build_general_summary_prompt(), role, temperature=0.15, max_tokens=300)
