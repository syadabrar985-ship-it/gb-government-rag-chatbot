
import os

import streamlit as st

# Configure the page
st.set_page_config(
    page_title="GB Government AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 GB Government AI Assistant")
st.write(
    "Ask questions about the Government of Gilgit-Baltistan."
)

# Configure the Groq API key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Import the pipeline
from rag_pipeline import RAGPipeline


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


# Load the RAG pipeline
try:
    pipeline = load_pipeline()

except Exception as error:
    st.error("The RAG pipeline could not be loaded.")
    st.exception(error)
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Receive a new question
question = st.chat_input(
    "Ask a question about Gilgit-Baltistan..."
)


if question:

    # Display the user's question
    st.chat_message("user").markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Generate the answer
    with st.chat_message("assistant"):

        with st.spinner("Searching and generating an answer..."):

            try:
                answer = pipeline.ask(question)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                st.error(
                    "An error occurred while generating the answer."
                )

                st.exception(error)
