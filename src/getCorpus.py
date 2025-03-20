
import os
import sys
import csv
import pandas as pd

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent)) 

def getCorpusSTF():
    
    temas = pd.read_csv('data/dataset_stf.csv', encoding='utf-8')
    
    ret = list(temas['Titulo'].values)
    
    return ret

def getCorpusSTJ():
    
    temas = pd.read_csv('data/dataset_stj_v2.csv', encoding='utf-8', delimiter=',')
    
    ## Retorna somente os temas que possuem titulo (nao vazio)
    tmp = temas.loc[temas['Titulo'].notnull()]
    
    ret = list(tmp['Titulo'].values)
    
    return ret

if __name__ == "__main__":
    
    print(getCorpusSTJ())