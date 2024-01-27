# torch packages
import torch

import src.model_registry as model_registry
import src.models.cv.alexnet.alexnet as alexnet
import src.models.cv.inception.inception_v1 as inception
import src.models.cv.lenet.lenet as lenet
import src.models.cv.mobilenet.mobilenet as mobilenet
import src.models.cv.resnet.resnet as resnet
import src.models.cv.vggnet.vgg as vgg


class ModelLoader:
    def __init__(self, model_name):
        """
        params: model_name -> from MLModel
        params: config -> dict containing input param to instantiate a model
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        model_list = [
            model_registry.MLModel.lenet.name,
            model_registry.MLModel.alexnet.name,
            model_registry.MLModel.vgg16.name,
            model_registry.MLModel.resnet50.name,
            model_registry.MLModel.inception.name,
            model_registry.MLModel.mobilenet.name,
        ]
        # Get input params and instantiate model
        if self.model_name.lower() == model_registry.MLModel.lenet.name:
            params = lenet.get_params()
            self.model = lenet.LeNet(params, device=self.device)
        elif self.model_name.lower() == model_registry.MLModel.alexnet.name:
            params = alexnet.get_params()
            self.model = alexnet.AlexNet(params, device=self.device)
        elif self.model_name.lower() == model_registry.MLModel.vgg16.name:
            params = vgg.get_params()
            self.model = vgg.VGG16(params, device=self.device)
        elif self.model_name.lower() == model_registry.MLModel.resnet50.name:
            params = resnet.get_params()
            self.model = resnet.ResNet50(params, device=self.device)
        elif self.model_name.lower() == model_registry.MLModel.inception.name:
            params = inception.get_params()
            self.model = inception.Inception_v1(params, device=self.device)
        elif self.model_name.lower() == model_registry.MLModel.mobilenet.name:
            params = mobilenet.get_params()
            self.model = mobilenet.MobileNet(params, device=self.device)

        else:
            raise AssertionError(
                f"model {model_name} is not registered in CV in model_registry pick from following list: {model_list}"
            )

        if self.model and torch.cuda.is_available():
            self.model.cuda()
