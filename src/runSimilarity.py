
import torch
import spacy
import csv
import sys
import time
import nltk

from csv import writer
from pathlib import Path

from nltk import sent_tokenize
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer 

from src.getCorpus import getCorpusSTJ, getCorpusSTF

from FlagEmbedding import BGEM3FlagModel

sys.path.append(str(Path(__file__).parent.parent.parent)) 

def similarityCompare_BGEM():
    
    # 1. Load a pretrained Sentence Transformer model
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)['dense_vecs']
    embeddings_2 = model.encode(sentences_2)['dense_vecs']
    
    similarity = embeddings_1 @ embeddings_2.T
    
    print(similarity)
    
    return ''

def similarityCompare(ret, modelo):
    
    try:
    
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer(modelo, similarity_fn_name=SimilarityFunction.COSINE)
        
        # Two lists of sentences
        temas = [
            'Tema 685 - Extensão da imunidade tributária recíproca ao IPVA de veículos adquiridos por município no regime da alienação fiduciária.',
            'Tema 243 - Termo inicial dos juros moratórios nas ações de repetição de indébito tributário.',
            'Tema 988 - Possibilidade de desoneração do estrangeiro com residência permanente no Brasil em relação às taxas cobradas para o processo de regularização migratória.',
            'Tema 246 - Responsabilidade subsidiária da Administração Pública por encargos trabalhistas gerados pelo inadimplemento de empresa prestadora de serviço.',
            'Tema 247 - Incidência do ISS sobre materiais empregados na construção civil.'
        ]
        
        # Compute embeddings for both lists
        embeddings1 = model.encode(temas)
        embeddings2 = model.encode(ret)

        # Compute cosine similarities
        similarities = model.similarity(embeddings1, embeddings2)     
                
        retorno = []
        
        for idx_j, sentence1 in enumerate(ret):
            
            for idx_i, sentence2 in enumerate(temas):
                # print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
                saida = f"{similarities[idx_i][idx_j]:.4f}"
                retorno.append(str(sentence2 + ' ' + saida))
                
            break
                
        return retorno
        
    except RuntimeError:        
        pass

def similarityTop(ret, modelo):    
    
    try:
                
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer(modelo, similarity_fn_name=SimilarityFunction.COSINE)

        # 2. Get the corpus
        corpus = getCorpusSTJ()
        
        # 3. Compute embeddings
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        
        # 4. Get Document 
        queries = [ret]
        
        # 5. Top 50 most similar sentences in corpus
        top_k = min(50, len(corpus))
        
        retorno = []        
        for query in queries:
        
            query_embedding = model.encode(query, convert_to_tensor=True)
            
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
            
            scores, indices = torch.topk(similarity_scores, k=top_k)

            for score, idx in zip(scores, indices):
                # print(corpus[idx], f"(Score: {score:.4f})")
                # retorno.append(str(corpus[idx] + ' (Score: ' + str(score) + ')'))
                retorno.append(str(corpus[idx]))
                
        return retorno
        
    except RuntimeError:        
        pass


if __name__ == '__main__':
    
    similarityCompare_BGEM()