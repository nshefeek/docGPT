from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "docGPT"
    ALLOWED_HOSTS: list = ["*"]
    DOCUMENT_STORE: str = "FAISS"
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()