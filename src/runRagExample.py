
# imports
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Get Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings
)

# Read in State of the Union Address File
with open("./data/2024_state_of_the_union.txt", encoding='utf-8') as f:
    state_of_the_union = f.read()

# Initialize Text Splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Create Documents (Chunks) From File
texts = text_splitter.create_documents([state_of_the_union])

# Save Document Chunks to Vector Store
ids = vector_store.add_documents(texts)

# Query the Vector Store
results = vector_store.similarity_search(
    'Who invaded Ukraine?',
    k=2
)

# Print Resulting Chunks
for res in results:
    print(f"* {res.page_content} [{res.metadata}]\n\n")
    
# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set Chroma as the Retriever
retriever = vector_store.as_retriever()

# Initialize the LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")

# Create the Prompt Template
prompt_template = """Use the context provided to answer the user's question below. If you do not know the answer based on the context provided, tell the user that you do not know the answer to their question based on the context provided and that you are sorry.
context: {context}

question: {query}

answer: """

# Create Prompt Instance from template
custom_rag_prompt = PromptTemplate.from_template(prompt_template)

# Create the RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Query the RAG Chain
print(rag_chain.invoke("According to the 2024 state of the union address, Who invaded Ukraine?"))

# Get an I don't know from the Model
print(rag_chain.invoke("What is the purpose of life?"))