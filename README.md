# 📊 Universal Agentic Data Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue )
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B )
![LangChain](https://img.shields.io/badge/LangChain-Agents-green )
![Groq](https://img.shields.io/badge/Groq-Qwen_27B-orange )
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED )

An autonomous, LLM-powered Data Analyst application that allows users to upload any CSV dataset and interact with it using natural language. Built with **LangChain's Pandas DataFrame Agent** and powered by **Groq's ultra-fast inference engine**, this agent can clean data, handle missing values, perform Exploratory Data Analysis (EDA), and generate interactive Plotly visualizations on the fly.

## 🚀 Live Demo
**[Click here to try the Live App on Streamlit Cloud](https://agentic-data-analyst-feoqpsnymptvuuv4c6wkaw.streamlit.app/ )**

## 🧠 Key Features
* **Agentic Workflow:** Uses a ReAct (Reasoning + Acting) loop to write, execute, and self-correct Python code in the background.
* **Universal CSV Support:** Dynamically analyzes any uploaded dataset without hardcoded schemas.
* **High-Speed Processing:** Utilizes `pyarrow` engine and `@st.cache_data` for instant data loading and memory optimization.
* **Interactive Visualizations:** Automatically generates and renders colorful, interactive `plotly.express` charts based on user prompts.
* **Developer Terminal:** Features a live LangChain callback terminal to view the AI's internal thought process and code execution.
* **One-Click Export:** Download the cleaned and processed dataset directly from the UI.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI Framework:** LangChain (Pandas Agent, StreamlitCallbackHandler)
* **LLM:** Groq API (Qwen-2.5-27B / Llama-3)
* **Data Processing:** Pandas, PyArrow
* **Visualization:** Plotly
* **MLOps:** Docker, GitHub Actions (CI/CD)

## 🐳 Run Locally via Docker (Recommended)
You can pull and run the pre-built Docker image directly from the GitHub Container Registry:

    # Pull the latest image
    docker pull ghcr.io/munnurumahesh03-coder/agentic-data-analyst:latest

    # Run the container
    docker run -p 8501:8501 ghcr.io/munnurumahesh03-coder/agentic-data-analyst:latest

*The app will be available at `http://localhost:8501`*

## 💻 Manual Local Installation

1. Clone the repository:
    git clone https://github.com/munnurumahesh03-coder/Agentic-Data-Analyst.git
    cd Agentic-Data-Analyst

2. Install dependencies:
    pip install -r requirements.txt

3. Run the application:
    streamlit run app.py

*(Note: You will need to enter your Groq API key in the app's sidebar to activate the AI ).*
