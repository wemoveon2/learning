from typing import List

from pydantic import BaseModel

class NerTextIn(BaseModel):
    text: str

class StanzaEntities(BaseModel):
    text: str
    type: str
    start_char: int
    end_char: int

class SentenceNerOutput(BaseModel):
    named_entities: List[StanzaEntities] 
    sentiment: str
    sentence: str

class NerOutput(BaseModel):
    sentences: List[SentenceNerOutput]
