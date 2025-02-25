import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from VietnameseEmbedding import embedding_model
from NongtriPrompt import reformulate_question_prompt_template, answer_prompt_template, general_chat_template, route_prompt
from formatdocs import format_docs

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load biến môi trường
load_dotenv()
cur_file_path = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(cur_file_path,"DB", "final_chroma_db_RecursiveCharacterTextSplitter")

# 2. Định nghĩa chat model và memory

non_creative_chat_model = ChatOpenAI(
    streaming=True,  # Enable streaming
    callbacks=[StreamingStdOutCallbackHandler()],  # Add the streaming callback
    model = "gpt-3.5-turbo", # show more models https://platform.openai.com/docs/pricing
    temperature=0
)

creative_chat_model = ChatOpenAI(
    model = "gpt-3.5-turbo", # show more models https://platform.openai.com/docs/pricing
    temperature=1
)

def get_memory():
    return ConversationBufferMemory(return_messages=True)

# 3. Khởi tạo ChromaDB và Retrever 
db = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path
)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)

def create_rag_chain(memory):

    # Define route classifier chain
    route_chain = (
        route_prompt
        | non_creative_chat_model.bind(max_tokens=20)
        | StrOutputParser()
        | (lambda x: "retrieval" if "retrieval" in x.lower() else "chat")
    )

    # Define general chat response chain
    general_chat_chain = (
        general_chat_template
        | creative_chat_model
        | StrOutputParser()
    )

    # Existing RAG components
    standalone_question_chain = (
        reformulate_question_prompt_template
        | creative_chat_model
        | StrOutputParser()
    )

    context_chain = (
        retriever
        | RunnableLambda(format_docs)
    )

    answer_retriever_chain = (
        answer_prompt_template
        | non_creative_chat_model
        | StrOutputParser()
    )

    # Full RAG chain
    rag_chain = (
        standalone_question_chain
        | RunnableParallel(
            context=context_chain,
            question=RunnablePassthrough(),
            chat_history=lambda x: memory.load_memory_variables({})["history"]
        )
        | answer_retriever_chain
    )

    # Complete chain with routing
    return RunnableBranch(
        (lambda x: route_chain.invoke(x) == "chat", general_chat_chain),
        rag_chain
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