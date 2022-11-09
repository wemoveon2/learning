from typing import List

from pydantic import BaseModel

class NerTextIn(BaseModel):
    text: str

class NerModelOut(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int
class NerTextOut(BaseModel):
    prediction: List[NerModelOut] 

