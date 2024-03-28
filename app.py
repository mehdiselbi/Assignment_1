import os
import tempfile
import streamlit as st
from streamlit_chat import message
from agent import Agent

#Load .env file
load_dotenv()

st.set_page_config(page_title="Assignment 1 ChatPDF Interactive Document Conversational Interface")


def display_messages():
    """
    Displays chat messages from the session state.
    
    Iterates through messages stored in `st.session_state["messages"]`, 
    displaying each using the `message` function from `streamlit_chat`. 
    Prepares an empty container for a loading spinner.
    """
    st.subheader("ChatPDF")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["spinner"] = st.empty()


def process_input():
    """
    Process user input from the Streamlit text input widget.
    
    This function checks if the user input is not empty and sends it to the agent for processing.
    The response from the agent is then appended to the session state messages list along with the user input.
    
    Inputs:
    None

    Outputs:
    None
    """
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["spinner"], st.spinner(f"Processing"):
            agent_text = st.session_state["agent"].ask(user_text)
        
        st.session_state["user_input"] = ""
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    """
    Read files uploaded to Streamlit and save them temporarily.
    
    For each file uploaded through the Streamlit file uploader widget, this function saves the file to a temporary location,
    then passes the file path to the agent for document ingestion. After processing, the temporary file is deleted.
    """
    st.session_state["agent"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["agent"].doc_load(file_path)
        os.remove(file_path)


def is_openai_api_key_set() -> bool:
    """
    Checks if the OpenAI API key is set in the session state.
    
    Returns:
        bool: True if the OpenAI API key is set and non-empty, False otherwise.
    """
    return len(st.session_state["OPENAI_API_KEY"]) > 0


def main():
    """
    Main function to initialize and run the Streamlit application.
    
    This function initializes the Streamlit session state, sets up the UI components (e.g., text input, file uploader),
    and processes user interactions with the application.
    """
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if is_openai_api_key_set():
            st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])
        else:
            st.session_state["agent"] = None

    st.header("Assignment 1 ChatPDF")
    
    st.subheader("Upload a PDF")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Ask your PDF", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)

    st.divider()


if __name__ == "__main__":
    main()
