import streamlit as st
import os
import time


from langchain_ollama import ChatOllama
#from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

#qwen2.5:latest 
## Load the GROQ API key
#os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
#os.environ['OLLAMA_API_KEY'] = os.getenv('OLLAMA_API_KEY')

ollama_api_key = os.getenv('OLLAMA_API_KEY')

llm = ChatOllama(model="qwen2.5:latest", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

## Creating vectors
def create_vector_embedding():
    if 'vectors' not in st.session_state:
        embeddings = OllamaEmbeddings(model = "nomic-embed-text")    #Doing Embedding
        loader = PyPDFDirectoryLoader("research_papers")   #Data ingestion Technique
        docs = loader.load()  #Data loading
        text_splitter = RecursiveCharacterTextSplitter(chunk_size  =1000, chunk_overlap = 200)    #Splitting th etext into chunk
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)


st.title("Research Paper Q&A")

if st.button("Document Embedding"):
    with st.spinner("Embedding Documents..."):
        create_vector_embedding()
    st.success("Vector database is ready")

user_prompt = st.text_input("Enter your query from the research paper")


if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Click 'Document Embedding' first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever()

    start = time.perf_counter()
    docs = retriever.invoke(user_prompt)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context, "input": user_prompt})
    print(f"Response time: {time.perf_counter() - start:.2f}s")

    st.write(answer)

    with st.expander("Document similarity Search"):
        for doc in docs:
            st.write(doc.page_content)
            st.write("---")