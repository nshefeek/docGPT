from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    PROJECT_NAME: str = "docGPT"
    MODEL_PATH: str = "./model_cache/"
    ALLOWED_HOSTS: list = ["*"]
    LLM_MODEL: str
    EMBEDDING_MODEL: str
    UPLOADS_DIR: str
    DATABASE_NAME: str
    COLLECTION_NAME: str
    DOCUMENT_STORE_URI: str = "http://localhost:9200"
    CORS_ORIGINS: list = ["*"]
    API_PREFIX: str = "/api"

    class Config:
        env_file = ".env"


settings = Settings()