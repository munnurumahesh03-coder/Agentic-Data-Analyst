import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Universal Data Analyst", page_icon="📊", layout="wide")

st.title("📊 Universal Agentic Data Analyst")
st.markdown("Upload any CSV, ask the AI to clean it, analyze it, or draw **interactive Plotly charts**!")

# 👈 THE FIX: Securely loading the API key from Streamlit Secrets!
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ GROQ_API_KEY is missing! Please add it to your Streamlit Cloud Secrets.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file, engine="pyarrow")

with st.sidebar:
    st.header("⚙️ Developer Tools")
    with st.expander("🖥️ Live AI Terminal", expanded=False):
        terminal_container = st.container()

uploaded_file = st.file_uploader("📂 Upload your messy CSV file", type=["csv"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("⚡ Processing file..."):
            st.session_state.df = load_data(uploaded_file)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = [] 
    
    st.success(f"✅ Loaded {uploaded_file.name} ({st.session_state.df.shape[0]} rows)")
    
    with st.expander("Preview Data"):
        st.dataframe(st.session_state.df.head())

    CUSTOM_PREFIX = """
    You are an elite Data Analyst. 
    CRITICAL RULES FOR ALL TASKS:
    1. The dataset is ALREADY loaded as `df`. DO NOT use pd.read_csv().
    2. NEVER print the entire dataframe. Always use `.sum()`, `.count()`, or `.head()`.
    3. MULTI-PART QUESTIONS: If the user asks for multiple things, write ONE single Python script that calculates all of them at once.
    
    SPECIFIC PROTOCOLS:
    - MISSING VALUES: Print `df.isnull().sum()`. If filling/dropping, use `inplace=True`.
    - DUPLICATES: Print `df.duplicated().sum()`. If dropping, use `df.drop_duplicates(inplace=True)`.
    - OUTLIERS: Calculate using IQR. ONLY print the COUNT of outliers per column.
    - PLOTS: Always check `df.columns` first. ALWAYS use `plotly.express`. ALWAYS use `template='plotly_white'` or `plotly_dark`. ALWAYS save as 'temp_plot.html' using `fig.write_html('temp_plot.html')`. Do NOT use plt.show().
    
    EXECUTION RULES (CRITICAL):
    - After you run the python code and get the observation, YOU MUST STOP CALLING TOOLS.
    - You MUST write a final conversational response to the user containing the answer.
    - Even if the result is 0, empty, or None, you MUST reply to the user in the chat.
    """

    llm = ChatGroq(temperature=0, model_name="qwen/qwen3.8-27b", max_tokens=800)
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state.df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="openai-tools",
        prefix=CUSTOM_PREFIX,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=True
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask the AI to clean data, find outliers, or draw a Plotly chart..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        st_callback = StreamlitCallbackHandler(terminal_container, expand_new_thoughts=True)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 AI is analyzing and writing code..."):
                try:
                    response = agent.invoke({"input": prompt}, {"callbacks": [st_callback]})
                    answer = response["output"]
                    
                    if "Agent stopped due to max iterations" in answer:
                        if os.path.exists("temp_plot.html"):
                            answer = "✅ Plot generated successfully!"
                        else:
                            steps = response.get("intermediate_steps", [])
                            if steps:
                                last_observation = steps[-1][1]
                                answer = f"{last_observation}"
                            else:
                                answer = "Task completed. Let me know what you'd like to analyze next!"
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if os.path.exists("temp_plot.html"):
                    with open("temp_plot.html", "r", encoding="utf-8") as f:
                        html_data = f.read()
                    components.html(html_data, height=500, scrolling=True)
                    os.remove("temp_plot.html")

    with st.sidebar:
        st.divider()
        st.header("📥 Export Data")
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Cleaned CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")
