import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Phase 1 imports
import config
from retriever import build_bm25_index, hybrid_search
from reranker import load_reranker, rerank

# Load environment variables
load_dotenv()

def process_pdf(uploaded_file):
    """Reads PDF and returns raw text."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def create_vector_store(text):
    """Splits text and creates a searchable database, plus a BM25 index."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using Google (with a workaround for batching bug)
    class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]
            
    embeddings = SafeGoogleEmbeddings(model="models/gemini-embedding-2")
    
    # Store in Chroma (in-memory for this example)
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    
    # Phase 1: Build BM25 index
    bm25_index = build_bm25_index(chunks)
    
    return vectorstore, bm25_index, chunks

def get_answer(vectorstore, bm25_index, chunks, user_query, chat_history="", 
               use_hybrid=config.HYBRID_SEARCH_ENABLED, 
               use_reranking=config.RERANKING_ENABLED, 
               alpha=config.HYBRID_ALPHA):
    """Searches the DB and asks Gemini for the answer."""
    # 1. Retrieve relevant text chunks
    if use_hybrid:
        retrieved_chunks = hybrid_search(vectorstore, bm25_index, chunks, user_query, 
                                         top_k=config.RETRIEVAL_TOP_K, alpha=alpha)
    else:
        docs = vectorstore.similarity_search(user_query, k=config.RETRIEVAL_TOP_K)
        retrieved_chunks = [doc.page_content for doc in docs]
        
    # Phase 1: Reranking
    source_chunks_with_scores = []
    if use_reranking and retrieved_chunks:
        reranker_model = load_reranker(config.RERANKER_MODEL)
        scored_chunks = rerank(reranker_model, user_query, retrieved_chunks, top_n=config.RERANKER_TOP_N)
        final_chunks = [chunk for chunk, score in scored_chunks]
        source_chunks_with_scores = scored_chunks
    else:
        final_chunks = retrieved_chunks[:config.RERANKER_TOP_N]
        # Just assign dummy scores if reranking is off
        source_chunks_with_scores = [(chunk, 0.0) for chunk in final_chunks]
    
    # 2. Initialize Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 3. Setup the prompt and chain
    prompt_template = """Use the following pieces of context and the chat history to answer the user's question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Helpful Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    chain = prompt | llm | StrOutputParser()
    
    # 4. Generate answer
    context = "\n\n".join(final_chunks)
    response = chain.invoke({"chat_history": chat_history, "context": context, "question": user_query})
    
    return {
        "answer": response,
        "source_chunks": source_chunks_with_scores
    }