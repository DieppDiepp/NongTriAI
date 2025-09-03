import os

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from VietnameseEmbedding import embedding_model
from NongtriPrompt import reformulate_question_prompt_template, answer_prompt_template, general_chat_template, route_prompt, routing_db_prompt, multi_query_prompt
from formatdocs import format_docs, get_unique_union, split_queries

# Ép llm chỉ được trả về ouput xác định 
from structured_response_llm import RouteClassification, PlantClassification

from mapping import PLANT_TYPE_MAPPING

# ignore warning of torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load biến môi trường
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  
cur_file_path = os.path.dirname(os.path.abspath(__file__))

db_path_caphe = os.path.join(cur_file_path,"DB", "VectorStore-db", "caphe")
db_path_tieu = os.path.join(cur_file_path,"DB", "VectorStore-db", "hotieu")
db_path_saurieng = os.path.join(cur_file_path,"DB", "VectorStore-db", "saurieng")

# 2. Định nghĩa chat model và memory
non_creative_chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

creative_chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1
)
# Đảm bảo llm chỉ trả về chat hoặc retrieval
routing_chat_or_retrievel_chat_model = non_creative_chat_model.with_structured_output(RouteClassification)
routing_db_structured_chat_model = creative_chat_model.with_structured_output(PlantClassification)

# 3. Khởi tạo ChromaDB và Retrever 
db_caphe = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path_caphe
)

db_tieu = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path_tieu
)

db_saurieng = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path_saurieng
)

retriever_caphe = db_caphe.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}
)

retriever_hotieu = db_tieu.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}
)

retriever_saurieng = db_saurieng.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}
)

def get_memory():
    return ConversationBufferMemory(return_messages=True)

# Full pipeline xử lý 
def create_rag_chain(memory):
    # Helper functions
    def format_context(input_dict: dict) -> list:
        '''
        Agrs:
            nhận vào 1 input dictionary từ create_context_chain,
            lấy các tài liệu trong value của khóa context_chain của dict create_context_chain
            sau đó format nó lại - xem hàm format_docs và get_unique_union kỹ hơn trong file import
        Returns:
            Như trên
        '''
        return format_docs(get_unique_union(input_dict["context_chain"]))

    def get_chat_history(_: dict) -> str:
        '''
        Args:
            nhận vào 1 input dict
        Returns:
            Trả về lịch sử hội thoại 
        '''
        return memory.load_memory_variables({})["history"]

    def retrieve_documents(plant_type: str, query: str) -> list:
        '''
        Args:
            Nhận vào 2 input:
            plant_type: string - loại cây trồng xác định được tạo ra bởi structured LLm
            query: string - câu hỏi dùng để retrievel 
        Returns:
            Trả về một iterater danh sách các chunk tài liệu liên quan tới query
        '''
        if plant_type == "caphe":
            return retriever_caphe.invoke(query)
        elif plant_type == "hotieu":
            return retriever_hotieu.invoke(query)
        elif plant_type == "saurieng":
            return retriever_saurieng.invoke(query)\
            
    def create_context_chain(input_dict: dict) -> dict:
        '''
    Tạo chuỗi context bằng cách kết hợp thông tin loại cây trồng và các truy vấn
    Xem tại full_routing_db_chain - parallel trước đó trả về 1 cái dict 
    Args:
        input_dict (dict): Dictionary đầu vào chứa các thông tin: 
            - plant_type (str): Loại cây trồng dạng mã ('caphe', 'hotieu', 'saurieng')
            - queries (List[str]): Danh sách các truy vấn đã được mở rộng
            - question (str): Câu hỏi gốc của người dùng
            - chat_history (List[Union[HumanMessage, AIMessage]]): Lịch sử hội thoại
            
    Returns:
        dict: Dictionary kết quả chứa:
            - context_chain (List[Document]): Danh sách tài liệu liên quan đã được lọc và định dạng
            - question (str): Giữ nguyên câu hỏi gốc từ input
            - chat_history (List[Union[HumanMessage, AIMessage]]): Giữ nguyên lịch sử hội thoại từ input
    
    Quy trình:
        1. Chuẩn hóa tên cây trồng sang tiếng Việt
        2. Kết hợp tên cây tiếng Việt vào từng truy vấn
        3. Truy xuất tài liệu từ database tương ứng
    '''
        plant_type = input_dict["plant_type"]
        # Chuẩn hóa tên cây trồng
        vietnamese_name = PLANT_TYPE_MAPPING.get(plant_type, plant_type)
        
        return {
            "context_chain": [
                doc
                for query in input_dict["queries"]
                # Thêm tên cây trồng tiếng Việt vào query
                for doc in retrieve_documents(plant_type, f"{query}, loại cây: {vietnamese_name}")
            ],
            "question": f"{input_dict['question']} [Loại cây: {vietnamese_name}]",
            "chat_history": input_dict["chat_history"]
        }

    # Chain definitions
    route_chain = (
        route_prompt
        | routing_chat_or_retrievel_chat_model
        | RunnableLambda(lambda x: x.datasource)  # Trích xuất giá trị từ Pydantic model
    )

    general_chat_chain = (
        general_chat_template
        | creative_chat_model
        | StrOutputParser()
    )

    answer_retriever_chain = (
        answer_prompt_template
        | non_creative_chat_model
        | StrOutputParser()
    )

    routing_db_chain = (
        RunnableLambda(lambda x: {
            "question": x["question"]["question"] if isinstance(x["question"], dict) else x["question"],
            "chat_history": x.get("chat_history", [])
        })
        | routing_db_prompt
        | routing_db_structured_chat_model
        | RunnableLambda(lambda x: x.plant_type)
    )

    multi_query_chain = (
        RunnableLambda(lambda x: x["question"])  # Trích xuất giá trị 'question' từ input dict
        | multi_query_prompt
        | creative_chat_model
        | StrOutputParser()
        | RunnableLambda(split_queries)
    )

    # Main RAG chain
    rag_chain = RunnableParallel(
        context=RunnableLambda(format_context),
        question=RunnablePassthrough(),
        chat_history=lambda x: x.get("chat_history", [])
    ) | answer_retriever_chain

    # Database routing chain
    full_routing_db_chain = (
        RunnableParallel(
            chat_history=RunnableLambda(get_chat_history),
            question=RunnablePassthrough()
        )
        | RunnableParallel(
            queries=multi_query_chain,
            plant_type=routing_db_chain,  # Nhận cả question và chat_history
            question=RunnablePassthrough(),
            chat_history=lambda x: x.get("chat_history", [])
        )
        | RunnableLambda(create_context_chain)
        | rag_chain
    )

    # Final chain with routing
    return RunnableBranch(
        (lambda x: route_chain.invoke(x) == "chat", general_chat_chain),
        full_routing_db_chain
    )

def continue_chat():
    print("Start chat with Nông Trí AI")
    memory = get_memory()
    chat_history = memory.load_memory_variables({})["history"]
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        rag_chain = create_rag_chain(memory)
        res = rag_chain.invoke({"question": query, "chat_history": chat_history})
        print(f"AI: {res}")

        memory.save_context({"input": query}, {"output": str(res)})
  

if __name__ == "__main__":
    # pass
    continue_chat()
