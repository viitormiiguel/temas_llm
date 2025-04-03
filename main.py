# -*- coding: utf-8 -*-
import os
import sys
import time
import openai
import anthropic
import streamlit as st

from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.globals import set_debug, set_verbose

from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent

from src.runLLM import load_prompt, load_llm
from src.parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from src.runSimilarity import similarityCompare, similarityTop, similarityTopBGE

# set_debug(True)
set_verbose(True)

sys.path.append(str(Path(__file__).parent.parent.parent)) 

# Connect to OpenAI GPT-3, fetch API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")


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
    
        if os.path.isfile(file) and file.endswith((".pdf", ".html")):
            print('Deleting file:', file)
            os.remove(file)
            
def extract_data():
    
    text_chunks = []
    
    files = filter(lambda f: f.lower().endswith((".pdf", ".html")), os.listdir("uploaded"))
        
    file_list = list(files)    
    
    for file in file_list:
        
        if '.pdf' in file:
        
            loader = PyPDFLoader(os.path.join('uploaded', file))
                   
            text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
                chunk_size = 512,
                chunk_overlap = 30,
                length_function = len,
                separators= ["\n\n", "\n", ".", " "]
            ))
            
        if '.html' in file:          
            
            loader = getContentAllHtml('uploaded/' + file)
            
            print(loader)
                        
            c_splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=0, separator=" ")
            
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450, 
                chunk_overlap=0, 
                separators=["\n\n", "\n", " ", ""]
            )
            
            pages = c_splitter.split_text(loader)
                        
            text_chunks = r_splitter.create_documents(pages)

    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    
    return vectorstore
            
def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)

with st.sidebar:
    
    st.subheader('')
    
    st.markdown('Aplicação do Processamento de Linguagem Natural na Análise de Similaridades de Temas STF/STJ'
        'Baseado em um texto de um recurso ou acórdão, quais os temas do STF/STJ possuem similaridade com os textos do processo.')

if __name__ == '__main__':
    
    ## Load LLM
    llm = load_llm()
    
    ## Load Prompt
    prompt = load_prompt()
    
    st.title("📝 Similaridades de Temas")
    
    retRag = []
    ragString = ''
    
    ## Section Documento
    with st.form("ragExec"):        
        
        uploaded_file = st.file_uploader(label="Faça o Upload do arquivo (Processo ou Recurso):", accept_multiple_files=True, type=["html", "pdf"])
                
        css = r'''
            <style>
                [data-testid="stForm"] {border: 0px}
            </style>
        '''

        st.markdown(css, unsafe_allow_html=True)
        
        # Section Run LLM
        if uploaded_file != []:
            
            initialize_session_state()
            
            for pdf in uploaded_file:
                
                save_uploadedfile(pdf)
            
            ## Extrai dados e convert chunks em embeddings    
            st.session_state.knowledge_base = extract_data()
            
            if '.pdf' in uploaded_file[0]:                
                ## Get PDF Content
                retPDF = getContentPdf('uploaded/' + uploaded_file[0].name)
            else:
                retPDF = getContentAllHtml('uploaded/' + uploaded_file[0].name)
            
            ## Busca 50 temas mais similares
            # retSimi = similarityTop(retPDF, 'distiluse-base-multilingual-cased-v2')
            # retSimi = similarityTop(retPDF, 'paraphrase-multilingual-mpnet-base-v2')
            retSimi = similarityTopBGE(retPDF, 'BAAI/bge-m3')
            
            retRag.append(retSimi)
            
            for rr in retRag[0]:
                ragString += rr + '\n\n'            
            
            ## Remove files from uploaded folder
            remove_files()
        
        ## Section Infos Prompt
        with st.expander("**INFORMAÇÕES DE PROMPT**"):
            
            st.markdown("**Especificação**")
            
            st.markdown('''
                Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, 
                que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
                Use os parametros abaixo para recuperar o contexto para a resposta.
            ''')
            
            st.markdown("**Contexto de Documento**")
            
            st.markdown(''' 
                Com base na lista de temas do STF/STJ abaixo, analise o seguinte documento e identifique a quais temas ele mais se assemelha. 
                Considere a relação de conteúdo, jurisprudência aplicável e palavras-chave presentes no texto. 
                Dentre os Temas Similares abaixo, liste os Temas mais relevantes e explique brevemente o motivo da correspondência. 
            ''')
            
            st.markdown("**50 Temas mais Similares:**")
            
            ## Printa os 50 temas mais similares
            if retRag:
                for rr in retRag[0]:
                    st.markdown('- ' + rr + '\n\n')
    
        ## Inicio da construção do Prompt
        question = f"Com base na lista de temas do STF/STJ abaixo, analise o seguinte documento e identifique a quais temas ele mais se assemelha. Considere a relação de conteúdo, jurisprudência aplicável e palavras-chave presentes no texto. Dentre os Temas Similares abaixo, liste os Temas mais relevantes e explique brevemente o motivo da correspondência. \n\nTemas Similares: \n\n{ragString}"
        
        send_button = st.form_submit_button("Processar Documento")
            
        if send_button:
            
            ## Colocar string inteira do retorno dos teams
            queryTemas = question + ragString
            
            try:
                
                callbacks = [LoggingHandler()]
                
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
                # response = rag_chain.invoke(queryTemas, config={"callbacks": callbacks})
                
                tools = []
                
                ## Agent 
                agent = create_structured_chat_agent(
                    llm=llm,
                    tools=tools,
                )
                
                agentExecutor = AgentExecutor(
                    agent=agent,
                    tools=tools   
                )
                                
                ## Imprime Resposta da LLM
                st.write(response)          

            except:
                
                alert = st.warning("Por favor, Realize o Upload de um arquivo que deseja realizar uma pergunta.", icon="🚨")
            