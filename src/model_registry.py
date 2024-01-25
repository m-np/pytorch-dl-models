import time
import os
import enum
import json
from src.models.nlp.load_model import ModelLoader as NLPModelLoader

class MLModel(enum.Enum):
    bertlm = 0
    transformer = 1

def get_model(model_name):   
    loader = NLPModelLoader(model_name)
    return loader