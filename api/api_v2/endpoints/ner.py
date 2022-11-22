import re 

import stanza
from stanza.pipeline.core import DownloadMethod
from fastapi import APIRouter

from ..schemas import ner_text

ner = stanza.Pipeline(processors = "tokenize,ner,sentiment", 
                      download_method = DownloadMethod.REUSE_RESOURCES, 
                      model_dir = "./stanza/", 
                      use_gpu = False)
truecaser = stanza.Pipeline(processors = "tokenize,pos", 
                           download_method = DownloadMethod.REUSE_RESOURCES,
                           model_dir = "./stanza/",
                           use_gpu = False)

UPPER = {"PROPN", "NNS"}
def truecase(text: str):
    doc = truecaser(text)
    normalized_sentences = [w.text.capitalize() if w.upos in UPPER else w.text for sent in doc.sentences for w in sent.words]
    return re.sub(" (?=[\.,'!?:;])", "", ' '.join(normalized_sentences))
    

SENTIMENT = {
        0: "negative",
        1: "neutral",
        2: "positive"
        }
router = APIRouter()

@router.post("/", response_model = ner_text.NerOutput)
async def run_ner_on_text(text: ner_text.NerTextIn):
    cased_text = truecase(text.text)
    doc = ner(cased_text)
    sentences = doc.sentences
    result = []
    for sentence in sentences:
        tmp = dict()
        tmp["named_entities"] = [ent.to_dict() for ent in sentence.ents]
        tmp["sentiment"] = SENTIMENT[sentence.sentiment]
        tmp["sentence"] = sentence.text
        result.append(tmp)
    print(result)
    return {"sentences" : result}

