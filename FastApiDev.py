from dotenv import load_dotenv
load_dotenv()
from main_processing import create_rag_chain, get_memory
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
# Bỏ import langserve

app = FastAPI(
    title="Nông Trí AI",
    version="1.0",
    description="Chatbot hỗ trợ nông nghiệp",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc danh sách domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary để lưu trữ các session chat
sessions = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Xử lý session
    if not request.session_id or request.session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = get_memory()
    else:
        session_id = request.session_id
    
    memory = sessions[session_id]
    chat_history = memory.load_memory_variables({})["history"]
    
    # Thực thi RAG chain
    rag_chain = create_rag_chain(memory)
    res = rag_chain.invoke({
        "question": request.question,
        "chat_history": chat_history
    })
    
    # Cập nhật bộ nhớ
    memory.save_context(
        {"input": request.question},
        {"output": str(res)}
    )
    
    return {"response": res, "session_id": session_id}

# XÓA PHẦN add_routes

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
