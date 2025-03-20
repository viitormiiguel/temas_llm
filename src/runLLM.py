
import sys
import os
import warnings

import streamlit as sl
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

from src.getCorpus import getCorpusSTJ, getCorpusSTF

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_prompt():
        
    # prompt = """Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, que busca assinalar os teses com ou sem repercussao geral (temas) do STF ou STJ mais relevantes de cada processo.
    #     Lista ordenadamente por relevencia as teses ou temas com ou sem repercussão geral mais relevantes em portugues.
    # """    
    
    # prompt = """ Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, 
    #     que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
    #     Contexto = {context}
    #     Pergunta = {question}
    # """
    
    # prompt = """
    #     Voce é um software especialista em assuntos juridicos, 
    #     focado em analise de processos e recursos, 
    #     que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
    #     Caso você não tenha informações relevantes, retorne 'Desculpe não consegui achar uma resolução para a sua questão".
    #     Use os parametros abaixo para recuperar o contexto para a resposta.
    #     Contexto = {context}
    #     Pergunta = {question}
    # """
    
    prompt = """
        Voce é um software especialista em assuntos juridicos, 
        focado em analise de processos e recursos, 
        que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
        Use os parametros abaixo para recuperar o contexto para a resposta.
        Contexto = {context}
        Pergunta = {question}
    """    
    
    prompt = ChatPromptTemplate.from_template(prompt)
    
    return prompt

def load_llm():
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    return llm

if __name__ == '__main__':    
       
    # call main function
    load_prompt()