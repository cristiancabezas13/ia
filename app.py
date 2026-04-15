import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Desmontando los LLMs con Groq",
    page_icon="🧠",
    layout="wide",
)


# =========================
# Utilidades generales
# =========================

def get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_groq_client() -> Groq | None:
    api_key = get_api_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_resource
def get_tokenizer():
    # tiktoken no trae el tokenizer exacto de todos los modelos abiertos,
    # así que usamos una codificación moderna y consistente para fines didácticos.
    return tiktoken.get_encoding("cl100k_base")


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def split_words(raw_text: str) -> List[str]:
    cleaned = raw_text.replace("\n", ",")
    words = [w.strip() for w in cleaned.split(",") if w.strip()]
    # eliminar duplicados preservando orden
    seen = set()
    unique_words = []
    for word in words:
        lower = word.lower()
        if lower not in seen:
            seen.add(lower)
            unique_words.append(word)
    return unique_words


# =========================
# Módulo 1: Tokenización
# =========================

def tokenize_text(text: str) -> Tuple[List[str], List[int]]:
    enc = get_tokenizer()
    token_ids = enc.encode(text)
    tokens = [enc.decode([tok]) for tok in token_ids]
    return tokens, token_ids


def render_tokens_html(tokens: List[str]) -> str:
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3"]
    parts = []
    for i, tok in enumerate(tokens):
        safe_tok = (
            tok.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "\\n")
            .replace(" ", "·")
        )
        parts.append(
            f"<span style='background:{colors[i % len(colors)]}; padding:6px 8px; margin:4px;"
            f" border-radius:10px; display:inline-block; border:1px solid #cbd5e1;'>"
            f"{safe_tok}</span>"
        )
    return "".join(parts)


# =========================
# Módulo 2: Embeddings
# =========================

def get_embeddings(words: List[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(words)
    return np.array(vectors)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_embedding_plot(words: List[str], vectors: np.ndarray):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    df = pd.DataFrame({
        "word": words,
        "x": reduced[:, 0],
        "y": reduced[:, 1],
    })
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="word",
        hover_name="word",
        title="Embeddings proyectados a 2D con PCA",
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    fig.update_layout(height=600)
    return fig, df


def build_analogy_figure(words: List[str], vectors: np.ndarray):
    if len(words) < 4:
        return None, None

    a, b, c, d = words[:4]
    va, vb, vc, vd = vectors[:4]
    predicted = va - vb + vc
    similarity = cosine_similarity(predicted, vd)

    pca = PCA(n_components=2)
    all_vectors = np.vstack([vectors[:4], predicted.reshape(1, -1)])
    reduced = pca.fit_transform(all_vectors)

    labels = [a, b, c, d, f"predicción: {a}-{b}+{c}"]
    df = pd.DataFrame({
        "label": labels,
        "x": reduced[:, 0],
        "y": reduced[:, 1],
    })

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["label"],
            textposition="top center",
            marker=dict(size=14),
        )
    )

    fig.add_annotation(
        x=df.loc[4, "x"],
        y=df.loc[4, "y"],
        text="Vector estimado",
        showarrow=True,
        arrowhead=2,
    )

    fig.update_layout(
        title=f"Verificación visual de la analogía: {a} - {b} + {c} ≈ {d}",
        xaxis_title="Componente principal 1",
        yaxis_title="Componente principal 2",
        height=600,
    )

    return fig, similarity


# =========================
# Módulo 3 y 4: Groq
# =========================

DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "groq/compound-mini",
]


def call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
):
    client = get_groq_client()
    if client is None:
        raise ValueError("No se encontró GROQ_API_KEY.")

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        stream=False,
    )
    end = time.perf_counter()

    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    metrics = {
        "wall_time": end - start,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "prompt_time": getattr(usage, "prompt_time", None),
        "completion_time": getattr(usage, "completion_time", None),
        "queue_time": getattr(usage, "queue_time", None),
        "total_time": getattr(usage, "total_time", None),
    }

    completion_tokens = metrics["completion_tokens"] or 0
    completion_time = metrics["completion_time"] or 0

    if completion_tokens > 0 and completion_time > 0:
        metrics["time_per_token_ms"] = (completion_time / completion_tokens) * 1000
        metrics["throughput_tps"] = completion_tokens / completion_time
    else:
        metrics["time_per_token_ms"] = None
        metrics["throughput_tps"] = None

    return answer, metrics, response


def compare_temperatures(system_prompt: str, user_prompt: str, model: str, top_p: float, max_completion_tokens: int):
    low_answer, low_metrics, _ = call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.2,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    high_answer, high_metrics, _ = call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.9,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    return (low_answer, low_metrics), (high_answer, high_metrics)


# =========================
# Interfaz
# =========================

st.title("🧠 Taller Técnico: Desmontando los LLMs")
st.caption("Aplicación en Streamlit para explorar tokenización, embeddings, inferencia con Groq y métricas de desempeño.")

with st.sidebar:
    st.header("Configuración global")
    api_key_loaded = bool(get_api_key())
    st.success("GROQ_API_KEY detectada") if api_key_loaded else st.warning("Falta configurar GROQ_API_KEY")

    selected_model = st.selectbox(
        "Modelo Groq",
        options=DEFAULT_MODELS,
        index=0,
        help="Empieza con modelos pequeños o low cost y luego compara con modelos más grandes.",
    )
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.2, 0.05)
    top_p = st.slider("Top-P", 0.1, 1.0, 1.0, 0.05)
    max_completion_tokens = st.slider("Máx. tokens de salida", 64, 2048, 512, 64)

    st.markdown("---")
    st.subheader("Modelos recomendados")
    st.markdown(
        "- **llama-3.1-8b-instant**: rápido y barato\n"
        "- **openai/gpt-oss-20b**: buen balance costo/razonamiento\n"
        "- **llama-3.3-70b-versatile**: más potente, más costoso"
    )

    st.markdown("---")
    st.info(
        "Para desplegar en Streamlit Cloud, pega tu clave en **Secrets** como `GROQ_API_KEY=\"...\"`."
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Tokenizador",
        "2. Embeddings",
        "3. Groq API",
        "4. README y guía",
    ]
)

with tab1:
    st.subheader("Módulo 1: El laboratorio del tokenizador")
    input_text = st.text_area(
        "Ingresa un texto para analizar",
        value="Los LLM convierten texto en tokens antes de razonar.",
        height=140,
    )

    if input_text.strip():
        tokens, token_ids = tokenize_text(input_text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Caracteres", len(input_text))
        col2.metric("Número de tokens", len(tokens))
        ratio = len(input_text) / max(len(tokens), 1)
        col3.metric("Caracteres / token", f"{ratio:.2f}")

        st.markdown("### Texto dividido en tokens")
        st.markdown(render_tokens_html(tokens), unsafe_allow_html=True)

        token_df = pd.DataFrame({
            "posición": list(range(len(tokens))),
            "token": tokens,
            "token_id": token_ids,
        })
        st.markdown("### Mapeo token → ID")
        st.dataframe(token_df, use_container_width=True)

        st.markdown("### Interpretación")
        st.write(
            "La tokenización muestra que el modelo no procesa texto palabra por palabra, sino fragmentos. "
            "Por eso una palabra puede dividirse en varias piezas y eso afecta costo, contexto y latencia."
        )

with tab2:
    st.subheader("Módulo 2: Geometría de las palabras")
    st.write(
        "Este módulo usa un modelo de embeddings de HuggingFace para convertir palabras en vectores, "
        "luego aplica PCA para proyectarlos en 2D y graficarlos con Plotly."
    )

    default_words = "king, man, woman, queen"
    raw_words = st.text_area(
        "Lista de palabras separadas por comas",
        value=default_words,
        height=100,
    )
    words = split_words(raw_words)

    if st.button("Generar embeddings y plano cartesiano"):
        if len(words) < 2:
            st.error("Ingresa al menos dos palabras.")
        else:
            with st.spinner("Calculando embeddings..."):
                vectors = get_embeddings(words)
                fig, df = build_embedding_plot(words, vectors)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

            if len(words) >= 4:
                analogy_fig, similarity = build_analogy_figure(words, vectors)
                if analogy_fig is not None:
                    st.markdown("### Reto de analogía vectorial")
                    st.plotly_chart(analogy_fig, use_container_width=True)
                    st.metric("Similitud coseno con el 4.º término", f"{similarity:.4f}")
                    st.write(
                        f"Interpretación: usando las primeras cuatro palabras como analogía, el sistema evalúa si "
                        f"**{words[0]} - {words[1]} + {words[2]} ≈ {words[3]}**. Cuanto más se acerque la similitud a 1, "
                        "más consistente es la relación vectorial."
                    )

with tab3:
    st.subheader("Módulo 3: Inferencia y razonamiento con Groq")
    system_prompt = st.text_area(
        "System prompt",
        value="Eres un asistente académico claro, técnico y breve.",
        height=100,
        help="Define el comportamiento global del modelo.",
    )
    user_prompt = st.text_area(
        "User prompt",
        value="Explica qué es la atención en transformers con un ejemplo sencillo.",
        height=140,
        help="Pregunta o instrucción específica del usuario.",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Enviar a Groq", type="primary"):
            if not user_prompt.strip():
                st.error("Escribe un user prompt.")
            elif not get_api_key():
                st.error("Falta la GROQ_API_KEY en variables de entorno o en st.secrets.")
            else:
                with st.spinner("Consultando Groq..."):
                    try:
                        answer, metrics, response = call_groq(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            model=selected_model,
                            temperature=temperature,
                            top_p=top_p,
                            max_completion_tokens=max_completion_tokens,
                        )
                        st.session_state["last_answer"] = answer
                        st.session_state["last_metrics"] = metrics
                        st.session_state["last_raw_response"] = response
                    except Exception as exc:
                        st.exception(exc)

    with col_b:
        if st.button("Comparar temp. 0.2 vs 0.9"):
            if not user_prompt.strip():
                st.error("Escribe un user prompt.")
            elif not get_api_key():
                st.error("Falta la GROQ_API_KEY en variables de entorno o en st.secrets.")
            else:
                with st.spinner("Ejecutando comparación..."):
                    try:
                        low, high = compare_temperatures(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            model=selected_model,
                            top_p=top_p,
                            max_completion_tokens=max_completion_tokens,
                        )
                        st.session_state["compare_low"] = low
                        st.session_state["compare_high"] = high
                    except Exception as exc:
                        st.exception(exc)

    st.markdown("### Qué deberías observar")
    st.write(
        "- **Temperatura baja (<0.3):** respuestas más repetibles y deterministas.\n"
        "- **Temperatura alta (>0.7):** respuestas más variadas y creativas.\n"
        "- **Top-P:** recorta la cola de probabilidad y controla cuánta diversidad acepta el modelo.\n"
        "- **System Prompt:** cambia el rol, tono y estilo global de la respuesta."
    )

    if "last_answer" in st.session_state:
        st.markdown("## Respuesta del modelo")
        st.write(st.session_state["last_answer"])

        metrics = st.session_state["last_metrics"]
        st.markdown("## Métricas de desempeño")
        m1, m2, m3 = st.columns(3)
        m4, m5, m6 = st.columns(3)

        m1.metric("Input tokens", metrics.get("prompt_tokens") or "-")
        m2.metric("Output tokens", metrics.get("completion_tokens") or "-")
        m3.metric("Total tokens", metrics.get("total_tokens") or "-")
        m4.metric("Time per token (ms)", f"{metrics['time_per_token_ms']:.2f}" if metrics.get("time_per_token_ms") else "-")
        m5.metric("Throughput (tokens/s)", f"{metrics['throughput_tps']:.2f}" if metrics.get("throughput_tps") else "-")
        m6.metric("Total time (s)", f"{metrics['total_time']:.4f}" if metrics.get("total_time") else f"{metrics['wall_time']:.4f}")

        with st.expander("Ver tiempos detallados"):
            detail_df = pd.DataFrame([
                {"métrica": "queue_time", "valor": metrics.get("queue_time")},
                {"métrica": "prompt_time", "valor": metrics.get("prompt_time")},
                {"métrica": "completion_time", "valor": metrics.get("completion_time")},
                {"métrica": "total_time", "valor": metrics.get("total_time")},
                {"métrica": "wall_time_local", "valor": metrics.get("wall_time")},
            ])
            st.dataframe(detail_df, use_container_width=True)

    if "compare_low" in st.session_state and "compare_high" in st.session_state:
        st.markdown("## Comparación de temperatura")
        (low_answer, low_metrics) = st.session_state["compare_low"]
        (high_answer, high_metrics) = st.session_state["compare_high"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Temperatura 0.2")
            st.write(low_answer)
            st.caption(
                f"tokens/s: {low_metrics['throughput_tps']:.2f}" if low_metrics.get("throughput_tps") else "Sin throughput"
            )
        with c2:
            st.markdown("### Temperatura 0.9")
            st.write(high_answer)
            st.caption(
                f"tokens/s: {high_metrics['throughput_tps']:.2f}" if high_metrics.get("throughput_tps") else "Sin throughput"
            )

with tab4:
    st.subheader("Guía para el README y despliegue")
    st.markdown(
        """
### Qué debe explicar tu README
1. **Objetivo del proyecto**: mostrar cómo funciona un LLM desde tokenización hasta generación.
2. **Tecnologías**: Streamlit, Groq, Plotly, PCA, tiktoken, SentenceTransformers.
3. **Cómo correrlo localmente**.
4. **Cómo desplegarlo en Streamlit Community Cloud**.
5. **Mini ensayo sobre self-attention**: explica cómo el modelo cambia la respuesta cuando cambias el contexto o el system prompt.

### Idea de análisis sobre self-attention
Cuando modificas el contexto, el modelo cambia qué tokens son más relevantes para construir la respuesta. Eso es una señal práctica de que la atención distribuye peso de manera distinta según el prompt. Por ejemplo, si al mismo user prompt le das un system prompt de “responde como profesor” y luego “responde como poeta”, la salida cambia porque el modelo atiende a instrucciones distintas dentro del contexto.

### Despliegue en Streamlit Cloud
- Sube este proyecto a GitHub.
- En Streamlit Community Cloud conecta el repositorio.
- Selecciona `app.py` como archivo principal.
- En **Secrets** pega:

```toml
GROQ_API_KEY = "tu_api_key_aqui"
```
        """
    )
