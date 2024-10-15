import os 
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #similarity search 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #responisble to convert chunks into vectors 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import sacrebleu

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Solve Financial Queries Q&A ChatBot")

llm = ChatGroq(groq_api_key = groq_api_key,model = "Gemma-7b-it" )
print(llm)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question 
<context>
{context}
<context>
Question : {input}

"""
)


def vector_embedding(): #reading docs in the form of pdf store it in chuncks
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data1")
        st.session_state.docs = st.session_state.loader.load() #document loading 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        #we are storing these in session state so that we can use this anytime, anywhere.
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

page_bg_image = '''
    <style>
    [data-testid="stApp"]{
    opacity: 0.9;
    background-image: url("https://i.pinimg.com/564x/9d/30/1d/9d301d5b1eb18d2e77c2b2ec174ae6df.jpg");
    }
    [data-testid="stHeader"]{
    background : rgb(255, 255, 255/ 56%);
    }
    </style>
    '''
st.markdown(page_bg_image,unsafe_allow_html=True)
#prompt1 = st.text_input("Enter your queries?")

st.write("**Please Load the documents before entering your queries**")
if st.button("**Load documents**"):
    vector_embedding()
    st.write("**Vector DB CREATED!**")

prompt1 = st.text_input("**Enter your queries**")

#if st.button("search"):
 #   vector_embedding()
 #   st.write("Searched")

import time 
if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever() #creates interface to give answers to queries 
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    st.write(start,"sec")
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    query_embedding = st.session_state.embeddings.embed_query(prompt1)
    doc_embeddings = st.session_state.vectors.index.reconstruct_n(0, len(st.session_state.final_documents))
    doc_texts = [doc.page_content for doc in st.session_state.final_documents]
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()

    with st.expander("Similarity search in terms of context"):
        for i,doc in enumerate(response["context"]):
            st.write(f"Document {i+1} (Similarity: {similarities[i]:.4f}):")
            st.write(doc.page_content)
            st.write("------------------------------------------")
# Calculate BLEU score
    reference_texts = [doc.page_content for doc in response["context"]]
    candidate_text = response['answer']
    bleu = sacrebleu.corpus_bleu([candidate_text], [reference_texts])
    st.write(f"BLEU Score: {bleu.score:.4f}")