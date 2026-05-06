import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    """Splits text and creates a searchable database."""
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
    return vectorstore

def get_answer(vectorstore, user_query):
    """Searches the DB and asks Gemini for the answer."""
    # 1. Retrieve relevant text chunks
    docs = vectorstore.similarity_search(user_query)
    
    # 2. Initialize Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 3. Setup the prompt and chain
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    
    # 4. Generate answer
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "question": user_query})
    return response