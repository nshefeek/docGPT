from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Untitled"
    ALLOWED_HOSTS: list = ["*"]
    DOCUMENT_STORE: str = "FAISS"
    OLLAMA_MODEL: str
    OLLAMA_URL: str = "http://localhost:11434"
    DATABASE_URI: str

    class Config:
        env_file = ".env"


settings = Settings()