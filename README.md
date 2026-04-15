# Taller Técnico: Desmontando los LLMs

Aplicación en **Streamlit** para visualizar:

1. **Tokenización** de texto.
2. **Embeddings** en un plano cartesiano 2D usando PCA.
3. **Inferencia con Groq API** variando temperatura, top-p, system prompt y user prompt.
4. **Métricas de desempeño** como input/output tokens, throughput y time per token.

---

## Estructura del proyecto

```bash
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets_example.toml
```

---

## Instalación local

### 1) Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3) Configurar la API Key de Groq

Crea el archivo `.streamlit/secrets.toml` con este contenido:

```toml
GROQ_API_KEY = "tu_api_key_aqui"
```

También puedes usar variable de entorno:

```bash
export GROQ_API_KEY="tu_api_key_aqui"
```

En Windows PowerShell:

```powershell
$env:GROQ_API_KEY="tu_api_key_aqui"
```

### 4) Ejecutar la app

```bash
streamlit run app.py
```

---

## Descripción de los módulos

### Módulo 1: Tokenizador
- Recibe texto libre.
- Lo divide en tokens con `tiktoken`.
- Muestra cada token con color alterno.
- Muestra el ID de cada token.
- Compara número de caracteres vs. número de tokens.

### Módulo 2: Geometría de Embeddings
- Recibe una lista de palabras.
- Usa un modelo de embeddings de HuggingFace (`all-MiniLM-L6-v2`).
- Reduce la dimensión con PCA a 2 componentes.
- Grafica los puntos en un plano cartesiano interactivo con Plotly.
- Incluye una comprobación visual de analogías vectoriales como:

```text
king - man + woman ≈ queen
```

### Módulo 3: Groq API
- Usa un modelo de Groq para responder prompts.
- Permite cambiar:
  - temperatura,
  - top-p,
  - max_completion_tokens,
  - system prompt,
  - user prompt.
- Permite comparar temperatura baja vs. alta para ver diferencias en creatividad y determinismo.

### Módulo 4: Métricas de desempeño
La app muestra:
- input tokens,
- output tokens,
- total tokens,
- total time,
- time per token (ms),
- throughput (tokens/s).

---

## Ensayo breve: cómo observé Self-Attention al cambiar el contexto

El concepto de **self-attention** no se ve directamente como una matriz dentro de esta app, pero sí se puede observar de forma práctica en el comportamiento del modelo. Cuando cambié el **system prompt**, por ejemplo de “eres un profesor técnico” a “eres un poeta creativo”, noté que la respuesta al mismo **user prompt** cambiaba tanto en el tono como en la estructura. Esto sugiere que el modelo no trata todos los tokens por igual, sino que asigna distinta relevancia a ciertas partes del contexto.

También observé que cuando el prompt del usuario incluía una instrucción específica, como “explica con ejemplo” o “responde en una tabla”, el modelo reorganizaba la salida de acuerdo con esa nueva prioridad. Desde la perspectiva de transformers, esto se puede interpretar como una redistribución de atención entre los tokens del contexto para producir el siguiente token más adecuado.

En otras palabras, el modelo responde diferente porque **atiende diferente**. El contexto anterior, las instrucciones globales y la pregunta puntual compiten por importancia dentro de la secuencia. Ese comportamiento fue visible en la app cuando comparé respuestas con el mismo tema, pero con distintos contextos de entrada.

---

## Modelos recomendados para probar en Groq

Puedes comenzar con:
- `llama-3.1-8b-instant`
- `openai/gpt-oss-20b`
- `llama-3.3-70b-versatile`

Estrategia sugerida:
- usar primero un modelo pequeño para velocidad y costo,
- luego comparar con uno mayor para calidad de razonamiento.

---

## Subir a GitHub

```bash
git init
git add .
git commit -m "Entrega inicial del taller LLM con Groq y Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

---

## Despliegue en Streamlit Community Cloud

1. Subir el proyecto a GitHub.
2. Entrar a Streamlit Community Cloud.
3. Conectar el repositorio.
4. Seleccionar `app.py` como archivo principal.
5. En **Secrets**, pegar:

```toml
GROQ_API_KEY = "tu_api_key_aqui"
```

6. Desplegar la app.

---

## Autor

Proyecto académico para el taller **Desmontando los LLMs**.
