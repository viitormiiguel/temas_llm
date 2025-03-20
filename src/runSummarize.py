
import tiktoken
import textwrap
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from time import monotonic

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from src.parserDoc import getContentAllHtml

sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

use_long_text = True

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

def summaryText(news_article):
    
	model_name = "gpt-4o-mini"

	text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model_name)

	texts 	= text_splitter.split_text(news_article)
	docs 	= [Document(page_content=t) for t in texts]

	## Initialize the model
	llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
 
	prompt_template = """Write a concise summary of the following, without miss the details of document:

		{text}

	CONSCISE SUMMARY IN PORTUGUESE:"""
 
	prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
 
	num_tokens = num_tokens_from_string(news_article, model_name)

	gpt_40_mini = 128000
	verbose 				= True

	if num_tokens < gpt_40_mini:
		chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
	else:
		chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose)

	start_time 	= monotonic()
	summary 	= chain.run(docs)
 
	print(f"Chain type: {chain.__class__.__name__}")
	print(f"Run time: {monotonic() - start_time}")
	print(f"Summary: {textwrap.fill(summary, width=100)}")
    
	return textwrap.fill(summary, width=100)

if __name__ == '__main__':
    
	ret = getContentAllHtml('52873158420238217000 - RELVOTO1.html')
  
	summaryText(ret)