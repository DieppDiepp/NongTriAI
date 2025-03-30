from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import numpy as np
from langchain.embeddings.base import Embeddings

model_name = 'dangvantuan/vietnamese-embedding'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu' #XPU 
model = model.to(device)

def get_embedding(text: str) -> List[float]:
    """
    Hàm lấy embedding cho một chuỗi text.
    Sử dụng tokenizer và model đã khởi tạo.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Lấy hidden states và thực hiện mean pooling theo attention mask
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return embedding.squeeze().cpu().numpy()

# Custom Embedding class cho LangChain
class PhoBERTEmbeddings(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = get_embedding(text)
            embeddings.append(emb.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
embedding_model = PhoBERTEmbeddings(model, tokenizer)
