from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from docgpt.api.routes import router
from docgpt.config import settings


app = FastAPI(title="Untitled")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)