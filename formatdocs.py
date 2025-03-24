from typing import List, Iterable
from langchain.schema import Document

def get_unique_union(documents: Iterable[Document]) -> List[Document]:
    """
    Lọc các Document trùng lặp dựa trên nội page_content
    
    Args:
        documents: Iterable chứa các Document cần lọc
        Nhận vào 1 dạng iterable (không phải là dạng list, chỉ cần có thể duyệt) các langchain retrievel document 
            Ví dụ: retriever_caphe.get_relevant_documents(query) trả về 
            "context_chain": [
                Document(page_content="Cà phê Arabica thích hợp khí hậu mát...", metadata={...}), -> mỗi cái này là 1 Document langchain retrievel
                Document(page_content="Kỹ thuật trồng sầu riêng...", metadata={...}),  -> mỗi cái này là 1 Document langchain retrievel
                # ... các documents khác
            ],
    Returns:
        Danh sách Document duy nhất, giữ nguyên thứ tự xuất hiện đầu tiên
    
    """

    seen = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs

def format_docs(docs: List[Document]) -> str: 
    '''
     Args:
        docs: Danh sách các Document cần định dạng
        Hàm nhận vào là 1 Document của langchain được retrievel 
        Kiểu Document có các thuộc tính như .page_content, metadata,...
            Ví dụ:
            Document(page_content="Cà phê Arabica thích hợp khí hậu mát...", metadata={...}), -> mỗi cái này là 1 Document langchain retrievel
    Returns:
        Chuỗi văn bản - string - đã được định dạng kèm metadata, các Document cách nhau bởi 2 dòng trống
        
    '''
    formatted = []
    for doc in docs:
        # Trích xuất metadata
        source_link = doc.metadata.get("source_link", "unknown")
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page_label", "unknown")
        # Format document kèm metadata
        formatted.append(
            f"Tài liệu: {doc.page_content}\n"
            f"Thông tin mô tả tài liệu: Trang: {page}, Tệp nguồn: {source_file}, Liên kết tệp Nguồn {source_link}\n"
        )
    return "\n\n".join(formatted)


def split_queries(output: str) -> list:
    '''
    Args:
        output: phản hồi từ llm, dạng string nhưng các câu hỏi được phân cách nhau bởi \n 
    Returns:
        Trả về dạng list để bước tiếp theo duyệt trong list lấy câu hỏi để retrievel
    '''
    return output.split("\n")
