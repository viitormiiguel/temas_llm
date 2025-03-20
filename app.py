# gpt3 professional email generator by stefanrmmr - version June 2022

import os
import sys
import openai
import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings

import streamlit.components.v1 as components

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from annotated_text import annotated_text
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from src.parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from src.runSimilarity import similarityCompare, similarityTop
from src.runSummarize import summaryText
from src.runLLM import load_prompt, load_llm

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent.parent)) 

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="Temas TJRS", page_icon="img/rephraise_logo.png",)

# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def summaryTextGPT3(text):
    
    ret = getContentAllHtml(text)
        
    r = summaryText(ret)
        
    return r

def callParser(arquivo, modelo):
    
    ret = getContentAllHtml(arquivo)
        
    r = similarityTop(ret, modelo)
        
    return r

def extract_data():
    
    text_chunks = []
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("uploaded"))
    file_list = list(files)
    
    for file in file_list:
        loader = PyPDFLoader(os.path.join('uploaded', file))
        text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 30,
            length_function = len,
            separators= ["\n\n", "\n", ".", " "]
        ))
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    
    return vectorstore

def initialize_session_state():
    
    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = None
        
def save_uploadedfile(uploadedfile):
    
    with open(os.path.join("uploaded", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def remove_files():
    
    path = os.path.join(os.getcwd(), 'uploaded')
    
    for file_name in os.listdir(path):
    
        file = os.path.join(path, file_name)
    
        if os.path.isfile(file) and file.endswith(".pdf"):
            print('Deleting file:', file)
            os.remove(file)
            
def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)


def header():
    
    st.image('img/logo.png')  

    ## Section Sidebar 
    with st.sidebar:
        
        st.subheader('Similaridade de Temas')
        
        st.markdown('Estudo e Aplicabilidade do Processamento de Linguagem Natural na Análise de Similaridades de Temas STF/STJ'
        'Baseado em um texto de um recurso ou acórdão, quais os temas do STF/STJ possuem similaridade com os textos do processo.'
        'O objetivo é verificar se a ação pode ficar sobrestada aguardando uma decisão superior.')
    
    st.write('\n')  
    
def sectionDataset():
    
    ## Section Dataset Corpus 
    st.subheader('\nTemas Corpus (Dataset)\n')

    st.markdown('Extração de dados do site oficial do STF e STJ, que consta informações de repercussão, descrição, título e tese dos temas. Dataset composto por 2685 registros (temas).')
    
    st.subheader('\nDataset STF (exemplo)\n')
    
    tab1, tab2 = st.tabs(["Dados STF", "Dados STJ"])
    
    with tab1:        
        df_stf = pd.read_csv("data/dataset_stf.csv")
        st.write(df_stf)
        
    with tab2:
        df_stj = pd.read_csv("data/dataset_stj_v2.csv", delimiter=",", on_bad_lines='skip')
        st.write(df_stj)
    

def sectionSimilaridades():
    
    st.subheader('\nAnalise de Similaridades\n')
    
    input_c1, input_c2, retornoRag = '','', ''
        
    retRag = []
        
    with st.form("parserFiles"):

        col1, col2 = st.columns(2)
        
        filesPdf = [ f for f in os.listdir('test') if f.endswith('.pdf') ]
        filesHtml = [ f for f in os.listdir('test') if f.endswith('.html') ]

        with col1:
            input_c1 = st.selectbox('Escolha o arquivo (Processo .html)', filesHtml, index=None, placeholder="Selecionar")
        
        with col2:
            input_c2 = st.selectbox('Escolha o arquivo (Recurso .pdf)', filesPdf, index=None, placeholder="Selecionar")
        
        col1, col2, col3 = st.columns([5, 5, 7])
        
        with col1:
            input_metric = st.selectbox('Escolha a metrica:', ('Cosine Similarity', 'Re-Rank', 'Cross-encoder'), index=0)
    
        with col2:
            input_model = st.selectbox('Escolha o modelo:', ('paraphrase-multilingual-MiniLM-L12-v2', 'distiluse-base-multilingual-cased-v2', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'multi-qa-distilbert-cos-v1'), index=0)

        with col3:
            
            st.write("\n")
            st.write("\n")
            
            ## Run Models
            submitted = st.form_submit_button("Run Model")
            
        if submitted:
            
            ## Method Run Models Similarity
            retorno = callParser(input_c1, input_model)   
            
            retRag.append(retorno)
            
            try:
                
                ## Section visualização            
                st.write('**Visualização de Documentos**')            

                tab1, tab2 = st.tabs(["Processo", "Recurso (.pdf)"])

                col1, col2 = st.columns(2)
                
                with tab1:

                    with open(f'test/{input_c1}','r', encoding='latin-1') as f: 
                        html_data = f.read()

                    # Show in webpage
                    st.components.v1.html(html_data, scrolling=True, height=900)

                with tab2:

                    pdf_viewer(f"test/{input_c2}", pages_to_render=[1])        
                        
            except FileNotFoundError:        
                pass;

            st.divider() 
            
            ## Section Results
            st.info('Top 50 Temas Similares:\n')
                        
            for r in retorno:
                
                html_str = f"""
                    <style> p.a {{font: 14px Arial;}} </style>
                    <p class="a"> {r} </p>
                """
                
                st.markdown(html_str, unsafe_allow_html=True)
            
                retornoRag += r + '\n'
            
            st.divider() 

            col1, col2 = st.columns(2)            
            
            ## Section PCA Result
            st.image("img/pca_exemplo.png") 
    
    return retornoRag

def main():
    
    ## Section Header
    header()

    ## Section Dataset
    sectionDataset()
    
    ## Section Similaridades
    retRag = sectionSimilaridades()
    
    ## Section RAG
    st.subheader('Prompt RAG\n')
            
    with st.form("ragExec", clear_on_submit=True):     
        
        pdf_docs = st.file_uploader(label="Faça o Upload do seu PDF:", accept_multiple_files=True, type=["pdf"])
        
        submitted = st.form_submit_button("Salvar Documento")
                
    # Section Run LLM
    if submitted and pdf_docs != []:
        
        initialize_session_state()
        
        for pdf in pdf_docs:
            save_uploadedfile(pdf)
            
        st.session_state.knowledge_base = extract_data()
        
        remove_files()
        
        pdf_docs = []
        
        alert = st.success(body=f"Realizado o Upload do PDF com Sucesso!", icon="✅")
        
        time.sleep(3)         
        alert.empty()    
    
    query = st.text_area(label='Faça uma pergunta sobre o documento:', height=250, value=f"Temas similares: {retRag}")
        
    if query:
        
        ## Colocar string inteira do retorno dos teams
        queryTemas = query + retRag
        
        try:
            similar_embeddings = st.session_state.knowledge_base.similarity_search(queryTemas)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
            
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
        
            response = rag_chain.invoke(queryTemas)
            st.write(response)          

        except:
            
            alert = st.warning("Por favor, Realize o Upload do PDF que Deseja Realizar Chat", icon="🚨")
            time.sleep(3)
            alert.empty()


if __name__ == '__main__':    
    
    ## Load LLM
    llm = load_llm()
    
    ## Load Prompt
    prompt = load_prompt()
       
    # call main function
    main()
