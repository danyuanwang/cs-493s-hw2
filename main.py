from train import train
import torch
from model import GPT
from model import GPTConfig
def main():
    #device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    #config = GPTConfig()
    #transformer = GPT(config).to(device_type)
    #transformer.load_state_dict(torch.load("number_model_3_layer_small.pth", weights_only= True, map_location=device_type))
    train("train.txt", 1, 0.001, [0.9, 0.98], model = None)  # Example training call with the loaded model


main()  