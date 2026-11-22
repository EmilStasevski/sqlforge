"""Streamlit demo over the gateway.  Run: uv run --group demo streamlit run src/sqlforge/serve/demo.py"""

from __future__ import annotations

import os

import requests
import streamlit as st

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8080")
EXAMPLES = {
    "weather": [
        "Wie viele Tage mit mehr als 20 mm Niederschlag gab es 2024 in Köln?",
        "Which city had the highest average daily maximum temperature in 2023?",
    ],
    "tpch": ["Wie viele Kunden gibt es pro Marktsegment?", "Show the 5 orders with the highest total price."],
    "smard": ["What was the average day-ahead price per month in 2025?"],
}

st.set_page_config(page_title="sqlforge", layout="wide")
st.title("sqlforge: Text-to-SQL (DE/EN)")
db = st.selectbox("Database", list(EXAMPLES))
question = st.text_input("Question", EXAMPLES[db][0])
if st.button("Ask") and question:
    r = requests.post(f"{GATEWAY}/ask", json={"question": question, "db": db}, timeout=120)
    data = r.json()
    st.caption(f"cache: {r.headers.get('X-Cache', '-')} · outcome: {data.get('outcome')}")
    st.code(data.get("sql") or "", language="sql")
    if data.get("rows"):
        st.dataframe([dict(zip(data["columns"], row, strict=False)) for row in data["rows"]])
    elif data.get("error"):
        st.error(data["error"])
