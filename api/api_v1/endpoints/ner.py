from fastapi import APIRouter
from transformers import (AutoTokenizer, 
                          AutoModelForTokenClassification,
                          pipeline)

from ..schemas import ner_text

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir = "./transformers/")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir = "./transformers/")
ner = pipeline("ner", aggregation_strategy="average", model = model, tokenizer = tokenizer)

router = APIRouter()

@router.post("/", response_model = ner_text.NerTextOut)
async def run_ner_on_text(body: ner_text.NerTextIn):
    results = ner(body.text)
    return {"prediction": results}

