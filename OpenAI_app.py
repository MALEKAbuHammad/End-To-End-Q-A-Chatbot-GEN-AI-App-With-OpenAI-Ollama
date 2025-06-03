import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OpenAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries accurately and concisely."),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, engine, temperature, max_tokens):
    try:
        llm = ChatOpenAI(
            model=engine,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Initialize session state for API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# Sidebar for settings
st.sidebar.title("Settings")
st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", value=st.session_state.api_key)

# Select the OpenAI model
engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:", placeholder="Type your question here...")

# Generate response only if API key and user input are provided
if user_input and st.session_state.api_key:
    with st.spinner("Generating response..."):
        response = generate_response(user_input, st.session_state.api_key, engine, temperature, max_tokens)
        st.write("**Response:**")
        st.write(response)
elif user_input:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
else:
    st.info("Please provide a question to get started.")