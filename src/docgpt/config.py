from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "docGPT"
    ALLOWED_HOSTS: list = ["*"]
    DOCUMENT_STORE: str = "FAISS"
    LLM_MODEL: str
    EMBEDDING_MODEL: str
    UPLOAD_DIR: str

    class Config:
        env_file = ".env"


settings = Settings()
