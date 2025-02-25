# import re

# def format_docs(docs):
#     """Clean newlines and join documents"""
#     cleaned = [re.sub('\\n{2,}', '\n', d.page_content).strip() for d in docs]
#     return '\n\n'.join(cleaned)

def format_docs(docs):
    formatted = []
    for doc in docs:
        # Trích xuất metadata
        source_link = doc.metadata.get("source_link", "unknown")
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page_label", "unknown")
        # Format document kèm metadata
        formatted.append(
            f"Content: {doc.page_content}\n"
            f"Source: trang: {page}, Tệp Nguồn: {source_file}, liên kết tệp Nguồn{source_link}"
        )
    return "\n\n".join(formatted)
