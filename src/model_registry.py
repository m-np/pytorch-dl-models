import enum

from src.models.audio.load_model import ModelLoader as AudioModelLoader
from src.models.cv.load_model import ModelLoader as CVModelLoader
from src.models.nlp.load_model import ModelLoader as NLPModelLoader


class MLModel(enum.Enum):
    bertlm = 0
    transformer = 1
    lenet = 3
    alexnet = 4
    vgg16 = 5
    resnet50 = 6
    inception = 7
    mobilenet = 8
    wavenet = 9
    tacotron2 = 10


class MLTask(enum.Enum):
    nlp = 0
    cv = 1
    audio = 2


def get_model(ml_task, model_name):
    loader = None
    if ml_task == MLTask.nlp.name:
        loader = NLPModelLoader(model_name)
    elif ml_task == MLTask.cv.name:
        loader = CVModelLoader(model_name)
    elif ml_task == MLTask.audio.name:
        loader = AudioModelLoader(model_name)

    return loader
