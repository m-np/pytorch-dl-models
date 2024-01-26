# torch packages
import torch
import src.models.cv.lenet.lenet as lenet
import src.model_registry as model_registry


class ModelLoader:
    def __init__(self, model_name):
        """
        params: model_name -> from MLModel
        params: config -> dict containing input param to instantiate a model
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None 
        model_list = [
                        model_registry.MLModel.lenet.name,
                      ]
        # Get input params and instantiate model
        if self.model_name.lower() == model_registry.MLModel.lenet.name:
            params = lenet.get_params()
            self.model = lenet.LeNet(
                            params, 
                            device=self.device)
            
        
        else:
            raise AssertionError (
                f"model {model_name} is not registered in CV in model_registry pick from following list: {model_list}")

        if self.model and torch.cuda.is_available():
            self.model.cuda()

