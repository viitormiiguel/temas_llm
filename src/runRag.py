
import tiktoken
import textwrap
import torch
import spacy
import csv
import time
import sys
import os
import streamlit as sl
import warnings
warnings.filterwarnings("ignore")

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) 

from parserDoc import getContentHtml, getContentAllHtml, getContentPdf
from getCorpus import getCorpusSTJ, getCorpusSTF

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

use_long_text = True   

def load_prompt():
    
    corpus = getCorpusSTF()
    
    processo = getContentAllHtml('52456291520238217000 - RELVOTO1.html')
    
    prompt = """ Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, 
        que busca assinalar os temas STF ou STJ mais relevantes de cada processo .
        Contexto = {dataset}
        Pergunta = {question} {processo}.
        Lista ordenadamente por relevencia os temas mais relevantes em portugues:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    
    return prompt


def load_llm():
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    return llm
            
def initialize_session_state():
    
    if "knowledge_base" not in sl.session_state:
        sl.session_state["knowledge_base"] = None
        
def format_docs(docs):
    
    return "\n\n".join(doc.page_content for doc in docs)
    
if __name__ == '__main__':
    
    query = 'Dentre os temas STF (Supremo Trtibunal Federal) disponiveis, realize uma analise e liste em ordem por revelencia, ao menos os 5, quais os temas que são similares ao texto do '
        
    llm     = load_llm()
    prompt  = load_prompt()    
    
    similar_embeddings = sl.session_state.knowledge_base.similarity_search(query)
    similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
    
    retriever = similar_embeddings.as_retriever()
    
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    response = rag_chain.invoke(query)

    print(response)