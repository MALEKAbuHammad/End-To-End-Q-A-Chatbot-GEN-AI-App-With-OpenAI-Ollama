import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries accurately and concisely."),
    ("user", "Question: {question}")
])

def generate_response(question, engine, temperature, max_tokens):
    try:
        llm = ChatOllama(
            model=engine,
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
st.title("Enhanced Q&A Chatbot With Ollama")

# Sidebar for settings
st.sidebar.title("Settings")

# Select the Ollama model
engine = st.sidebar.selectbox("Select Ollama model", ["llama3", "mistral", "gemma"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:", placeholder="Type your question here...")

# Generate response only if user input is provided
if user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input, engine, temperature, max_tokens)
        st.write("**Response:**")
        st.write(response)
else:
    st.info("Please provide a question to get started.")