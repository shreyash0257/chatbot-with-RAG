import streamlit as st
from rag_util import process_pdf, create_vector_store, get_answer

st.set_page_config(page_title="AI PDF Chat", page_icon="🤖")

st.title("🤖 AI Chatbot with RAG")
st.caption("Upload a PDF to give the AI context.")

with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing..."):
                raw_text = process_pdf(uploaded_file)
                st.session_state.vs = create_vector_store(raw_text)
                st.success("Ready to chat!")

st.header("Chat")
if "vs" in st.session_state:
    query = st.text_input("Ask a question about your document:")
    if query:
        with st.chat_message("assistant"):
            res = get_answer(st.session_state.vs, query)
            st.write(res)
else:
    st.info("Upload a document in the sidebar to begin.")