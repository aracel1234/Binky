from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.tfidf_search import TfidfRetriever
from app.llama_utils import generate_answer

retriever = TfidfRetriever("data_buku.csv")
app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_q(data: Question):
    docs = retriever.retrieve(data.query)
    context = " ".join(docs['sinopsis'].values.tolist())
    answer = generate_answer(context, data.query)
    return {"answer": answer}
