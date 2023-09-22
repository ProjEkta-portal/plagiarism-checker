from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]


model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )

def chunk(text, plagiarized_text):
  words = text.split(" ")
  num_words = len(words)
  chunks = []
  for i in range(0, num_words, 32):
    chunks.append(" ".join(words[i:i+40]))

  words = plagiarized_text.split(" ")
  num_words = len(words)
  pl_chunks = []
  for i in range(0, num_words, 32):
    pl_chunks.append(" ".join(words[i:i+40]))
  
  return chunks, pl_chunks

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/compare-texts/")
async def compare_texts(request: Request):
    try:
        data = await request.json()
        # text_pair = TextPair(data["text1"], data["text2"])
        similarity_score = calculate_similarity(data["text1"], data["text2"])
        print(similarity_score)
        return {"similarity_score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input data")

def calculate_similarity(text1, text2):
    # Implement your text similarity calculation logic here.
    # For example, you can use libraries like spaCy, NLTK, or scikit-learn to perform text analysis.
    # For this example, we'll use a simple character-level similarity metric.
    
    text1_chunks, text2_chunks = chunk(text1, text2)

    vectors_text1 = model.encode(text1_chunks)
    vectors_text2 = model.encode(text2_chunks)

    dot = vectors_text1 @ vectors_text2.T

    a_val = np.sqrt(np.sum(np.multiply(vectors_text1, vectors_text1), axis=1))
    b_val = np.sqrt(np.sum(np.multiply(vectors_text2, vectors_text2), axis=1))

    denom = a_val.reshape(-1,1) @ b_val.reshape(1,-1)
    
    cosine_similarity = np.divide(dot, denom)

    cosine_similarity = np.max(cosine_similarity, axis=0)
    
    return cosine_similarity.mean().item()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
