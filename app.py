import streamlit as st
from rag_util import process_pdf, create_vector_store, get_answer

st.set_page_config(page_title="AI PDF Chat", page_icon="🤖")

st.title("🤖 AI Chatbot with RAG")
st.caption("Upload a PDF to give the AI context.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing..."):
                raw_text = process_pdf(uploaded_file)
                st.session_state.vs = create_vector_store(raw_text)
                st.session_state.messages = [] # Clear history on new upload
                st.success("Ready to chat!")

st.header("Chat")
if "vs" in st.session_state:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your document:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Format chat history for the prompt
        chat_history_str = ""
        for msg in st.session_state.messages[:-1]: # exclude current prompt
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history_str += f"{role}: {msg['content']}\n"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = get_answer(st.session_state.vs, prompt, chat_history_str)
                st.markdown(res)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": res})

else:
    st.info("Upload a document in the sidebar to begin.")