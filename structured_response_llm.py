from pydantic import BaseModel, Field
from typing import Literal

# Đảm bảo llm chỉ trả về phản hồi đúng mong đợi
class RouteClassification(BaseModel):
    """Phân loại câu hỏi người dùng"""
    datasource: Literal["chat", "retrieval"] = Field(
        ...,
        description="Chọn 'chat' cho hội thoại thông thường, 'retrieval' cho câu hỏi cần truy vấn thông tin về cây trồng"
    )

# Đảm bảo llm chỉ trả về phản hồi đúng mong đợi
class PlantClassification(BaseModel):
    plant_type: Literal["hotieu", "caphe", "saurieng"] = Field(
        ...,
        description="Chỉ được chọn một trong: 'hotieu', 'caphe', 'saurieng'"
    )
