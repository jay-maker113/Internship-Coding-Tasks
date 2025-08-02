"""
Task 2 
Build a FastAPI to get data from a blog post and create its summary and also answer any query asked in the payload. Use any openly available LLM for answers from Huggingface.
Body
{
	“Url”: “https://www.totalhealthandfitness.com/nutrition-and-chronic-disease/”,
	“Query”: ”What are the Nutrients for Managing Specific Chronic Conditions ? “ 
}

Response
{	
	“Summary”:” …….”
	“Query_Response”:”................”
}
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import uvicorn
import torch

app = FastAPI() # object of FastAPI

class BlogRequest(BaseModel): #pydabtic model for request body; expects JSON input
    url: HttpUrl
    query: Optional[str] = None

summarizer = None # blog summarization model
qa_model = None # question-answering model

@app.on_event("startup") # load models on startup
def load_models():
    global summarizer, qa_model
    try:
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn" # BART model from huggingface for summarization
        )
        qa_model = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2" # RoBERTa model from huggingface for question answering
        )
    except Exception as e:
        print(f"Error loading models: {e}")

def get_article_text(url: str) -> str: #fetch and clean article text
    headers = {'User-Agent': 'Mozilla/5.0'} #http headers to mimic a browser request
    response = requests.get(url, headers=headers)
    response.raise_for_status() 
    
    soup = BeautifulSoup(response.content, 'html.parser') # parse HTML content; web scraping
    
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
    text = ' '.join([elem.get_text() for elem in text_elements])
    
    text = re.sub(r'\s+', ' ', text).strip() # clean up whitespace
    
    if not text:
        raise ValueError("Could not extract any text from the article.")
        
    return text

def make_summary(text: str) -> str: # summarization function
    try:
        text_to_summarize = text[:2000]
        summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except:
        return "Could not generate summary."

def answer_question(text: str, question: str) -> str: # question answering function
    try:
        result = qa_model(question=question, context=text)
        return result['answer']
    except:
        return "No answer found in the article."

@app.post("/process-blog") # api endpoint to process blog
def process_blog(request: BlogRequest):
    try:
        article_text = get_article_text(str(request.url))
        
        summary = make_summary(article_text)
        
        query_response = None
        if request.query:
            query_response = answer_question(article_text, request.query)
        
        return {
            "summary": summary,
            "query_response": query_response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")

@app.get("/")
def home():
    return {"message": "Blog Summarizer API - Send POST to /process-blog"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
