from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
import torch

embedder = SentenceTransformer('/sentence-transformers/paraphrase-distilroberta-base-v1', quantize_model=True) #Embedding model, will download if not exists
embedder.max_seq_length = 25
BATCH_SIZE = 4

#App object with default config
app = FastAPI()

#Define data model
class Articles(BaseModel):
    headlines: List[str] = []

#Health endpoint to test liveness
@app.get("/health")
async def health():
    return {"message": "We are live"}

#Main endpoint to read CSVs and process patient records
@app.post("/embeddings")
async def make_embeddings(articles: Articles):
    
    articles = dict(articles)
    _embeddings = embedder.encode(articles['headlines'], show_progress_bar=True, batch_size=BATCH_SIZE, \
                                                convert_to_tensor=True) #Calculate embeddings of partition articles
    _embeddings = torch.nn.functional.normalize( _embeddings, p=2, dim=1 )
    _embeddings = _embeddings.numpy().tolist()
    return {"embeddings": _embeddings}