# torch packages
import torch
import src.models.nlp.original_transformer.transformer as transformer
import src.models.nlp.Bert.bert as bert
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
        # Get input params and instantiate model
        if self.model_name.lower() == model_registry.MLModel.transformer.name:
            params = transformer.get_params()
            self.model = transformer.Transformer(
                            params, 
                            device=self.device)
            
        elif self.model_name.lower() == model_registry.MLModel.bertlm.name:
            params = bert.get_params()
            self.model = bert.BERTLM(
                            params, 
                            device=self.device)
        else:
            raise AssertionError (f"model {model_name} is not registered in model_registry")

        if self.model and torch.cuda.is_available():
            self.model.cuda()

