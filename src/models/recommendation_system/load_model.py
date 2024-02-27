# torch packages
import torch

import src.model_registry as model_registry
import src.models.recommendation_system.ripplenet.ripplenet as ripplenet


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
            model_registry.MLModel.ripplenet.name,
        ]
        # Get input params and instantiate model
        if self.model_name.lower() == model_registry.MLModel.ripplenet.name:
            params = ripplenet.get_params()
            self.model = ripplenet.RippleNet(params, device=self.device)

        else:
            raise AssertionError(
                f"model {model_name} is not registered in NLP in model_registry pick from following list: {model_list}"
            )

        if self.model and torch.cuda.is_available():
            self.model.cuda()