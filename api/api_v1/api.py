from fastapi import APIRouter

from .endpoints import ner


router = APIRouter()
router.include_router(ner.router, prefix = "/ner", tags = ["NER"])
