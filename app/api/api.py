from fastapi import APIRouter
from app.api.endpoints import home

api_router = APIRouter()

api_router.include_router(home.router)
