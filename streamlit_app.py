import streamlit as st
from langchain.llms import OpenAI

st.title('My First LangChain app')
openai_api_key = "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"


def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
