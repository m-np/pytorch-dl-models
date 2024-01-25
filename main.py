# torch packages
import torch
from src.models.nlp.original_transformer.transformer import Transformer
from src.models.nlp.Bert.bert import BERTLM
import config
import json
from utils.common import count_parameters

if __name__ == "__main__":
    """
    Following parameters are for Multi30K dataset
    """
    # # Load config containing model input parameters 
    # # params = config.transformer_params
    # with open("config/transformer_params.json") as json_data:
    #     config = json.load(json_data)
    # print(config)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # Instantiate model
    # model = Transformer(
    #                 config["dk"], 
    #                 config["dv"], 
    #                 config["h"],
    #                 config["src_vocab_size"],
    #                 config["target_vocab_size"],
    #                 config["num_encoders"],
    #                 config["num_decoders"],
    #                 config["dim_multiplier"], 
    #                 config["pdropout"],
    #                 device = device)
    # if torch.cuda.is_available():
    #     model.cuda()
    # print(model)
    # print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Load config containing model input parameters 
    # params = config.transformer_params
    with open("config/bert_params.json") as json_data:
        config = json.load(json_data)
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate model
    model = BERTLM(
                config["dk"], 
                config["dv"], 
                config["h"],
                config["vocab_size"],
                config["seq_len"],
                config["num_encoders"],
                config["dim_multiplier"], 
                config["pdropout"])
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')


