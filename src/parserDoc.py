
import os
import sys

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent)) 

def getContentHtml(arquivo):

    with open("test/" + arquivo) as fp:
        
        soup = BeautifulSoup(fp, 'html.parser')
        
        arPara, arCita1, arCita2 = ([] for i in range(3))

        ## Paragrafo Padrao
        for para in soup.find_all('p', {'class' : 'paragrafoPadrao'}):            
            arPara.append(para.text.strip())

        ## Texto Citacao
        for cita in soup.find_all('p', {'class' : 'citacao'}):
            arCita1.append(cita.text.strip())            
        
        ## Texto Citacao 2
        for cita in soup.find_all('p', {'class' : 'citacao2'}):            
            arCita2.append(cita.text.strip())
 
        return arPara, arCita1, arCita2
    
def getContentAllHtml(arquivo):
    
    all = ''
    
    with open(arquivo, encoding="utf-8", errors="ignore") as fp:
        
        soup = BeautifulSoup(fp, 'html.parser')
        
        ## Paragrafo Padrao
        for para in soup.find_all('p', {'class' : 'paragrafoPadrao'}):            
            all += ' ' + para.text.strip()

        ## Texto Citacao
        for cita in soup.find_all('p', {'class' : 'citacao'}):
            all += ' ' + cita.text.strip()          
        
        ## Texto Citacao 2
        for cita in soup.find_all('p', {'class' : 'citacao2'}):            
            all += ' ' + cita.text.strip()
    
    return all

def getContentPdf(processo):
    
    # Abre arquivo de Processo
    reader = PdfReader(processo)

    # printing number of pages in pdf file
    # print(len(reader.pages))
    
    arText = ''
    
    # getting a specific page from the pdf file
    for i in range(len(reader.pages)):
                
        page = reader.pages[i]

        # extracting text from page
        text = page.extract_text()
        
        arText += '\n' + text.strip()
    
    return arText

if __name__ == "__main__":
    
    getContentPdf('test/50708196120238217000-recurso.pdf')