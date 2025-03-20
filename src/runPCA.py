
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def runTemas(model, retTemas, n_doc):
    
    # Load model from HuggingFace Hub
    tokenizer   = AutoTokenizer.from_pretrained('sentence-transformers/'+ model)
    model       = AutoModel.from_pretrained('sentence-transformers/'+ model)
    
    # Sentences we want sentence embeddings for
    sentences = [ retTemas ]
    
    labels = ['Alvo', 'Tema 247','Tema 212','Tema 287','Tema 581','Tema 125']
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    for i in range(0, len(sentences)):
        print(
            sentences[0],
            sentences[i],
            (sentence_embeddings[0]@sentence_embeddings[i]).item()
        )  
        
    # Initialize PCA, reducing to 2 dimensions
    pca = PCA(n_components=2)

    # Fit and transform the embeddings
    embeddings_2d = pca.fit_transform(sentence_embeddings)

    # Plotting with labels for each data point
    plt.figure(figsize=(8, 6))
    
    for i, point in enumerate(embeddings_2d):
        
        plt.scatter(point[0], point[1], c='blue', edgecolor=None, s=100)
        plt.text(point[0] + 0.02, point[1] - 0.014, labels[i], fontsize=14)

    plt.title("2D PCA Projection of Embeddings")
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plt.gca().spines['top'].set_visible(False)    # Hide top border
    plt.gca().spines['right'].set_visible(False)    # Hide top border

    # plt.show()
    plt.savefig('img/pca/pca_' + str(n_doc) + '.png')
    
    return ''

if __name__ == '__main__':    
    
    runTemas()