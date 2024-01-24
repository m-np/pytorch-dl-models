# torch packages
import torch
from src.models.nlp.original_transformer import Transformer
import config
import json
from utils.common import count_parameters

if __name__ == "__main__":
    """
    Following parameters are for Multi30K dataset
    """
    # Load config containing model input parameters 
    # params = config.transformer_params
    with open("config/transformer_params.json") as json_data:
        config = json.load(json_data)
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate model
    model = Transformer(
                    config["dk"], 
                    config["dv"], 
                    config["h"],
                    config["src_vocab_size"],
                    config["target_vocab_size"],
                    config["num_encoders"],
                    config["num_decoders"],
                    config["dim_multiplier"], 
                    config["pdropout"],
                    device = device)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    