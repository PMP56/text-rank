
from typing import List


class Settings():
    VERISON: float = 0.1
    PROJECT_NAME: str = "TextRank"
    DESCRIPTION: str = "TextRank algorithm for keyword extraction and summarization"

    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:8000"]


core_settings = Settings()