import time
import os
import enum
import json
from src.models.nlp.load_model import ModelLoader as NLPModelLoader
from src.models.cv.load_model import ModelLoader as CVModelLoader

class MLModel(enum.Enum):
    bertlm = 0
    transformer = 1
    lenet = 3
    alexnet = 4
    vgg16 = 5
    resnet50 = 6
    inception = 7
    # TODO below models
    # mobilenet = 7

class MLTask(enum.Enum):
    nlp = 0
    cv = 1

def get_model(ml_task, model_name):   

    loader = None
    if ml_task == MLTask.nlp.name:
        loader = NLPModelLoader(model_name)
    elif ml_task == MLTask.cv.name:
        loader = CVModelLoader(model_name)
        
    return loader