import torch

from model import GPT
from model import GPTConfig 
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy 

def inference(model_weights, config):
    #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    transformer = GPT(config).to(device_type)
    transformer.load_state_dict(torch.load(model_weights, weights_only= True, map_location=device_type))

    #read data
    d_tokens = config.tokenizer("I love machine").input_ids
    d_tokens = d_tokens[:-1]  # Remove the CLS token
    pad = [0] * (config.block_size - len(d_tokens))
    d_tokens = pad + d_tokens  # Pad the input tokens
    x_tokens = torch.tensor(d_tokens, dtype = torch.long)
    x_tokens = torch.tensor(x_tokens).unsqueeze(0)  # Add batch dimension

    #run inference
    logits = transformer(x_tokens)
    print(logits.shape)
    predicted_token = logits.argmax(dim=-1).squeeze()
    print(f"Predicted token: {predicted_token}")
    predicted_text = config.tokenizer.decode(predicted_token)
    print(f"Input: {config.tokenizer.decode(d_tokens)}")    
    print(f"Predicted: {predicted_text}")

inference("GPT_simple_data", GPTConfig())