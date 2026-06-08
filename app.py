import streamlit as st
import config
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
                vs, bm25, chunks = create_vector_store(raw_text)
                st.session_state.vs = vs
                st.session_state.bm25_index = bm25
                st.session_state.chunks = chunks
                st.session_state.messages = [] # Clear history on new upload
                st.success("Ready to chat!")
                
    st.divider()
    st.header("Search Settings")
    search_mode = st.radio("Search Mode", ["Hybrid", "Dense Only"])
    use_hybrid = (search_mode == "Hybrid")
    use_reranking = st.checkbox("Reranking", value=config.RERANKING_ENABLED)
    alpha = st.slider("Hybrid Alpha (Dense vs Sparse)", min_value=0.0, max_value=1.0, value=config.HYBRID_ALPHA, step=0.1)

st.header("Chat")
if "vs" in st.session_state:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display retrieved sources if available
            if "source_chunks" in message and message["source_chunks"]:
                with st.expander("📎 Retrieved Sources"):
                    for i, (chunk, score) in enumerate(message["source_chunks"]):
                        st.markdown(f"**Chunk {i+1}** (Score: `{score:.4f}`)\n```text\n{chunk}\n```")

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
                response_data = get_answer(
                    st.session_state.vs, 
                    st.session_state.bm25_index, 
                    st.session_state.chunks, 
                    prompt, 
                    chat_history_str,
                    use_hybrid=use_hybrid,
                    use_reranking=use_reranking,
                    alpha=alpha
                )
                answer = response_data["answer"]
                source_chunks = response_data.get("source_chunks", [])
                
                st.markdown(answer)
                if source_chunks:
                    with st.expander("📎 Retrieved Sources"):
                        for i, (chunk, score) in enumerate(source_chunks):
                            st.markdown(f"**Chunk {i+1}** (Score: `{score:.4f}`)\n```text\n{chunk}\n```")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "source_chunks": source_chunks
        })

else:
    st.info("Upload a document in the sidebar to begin.")