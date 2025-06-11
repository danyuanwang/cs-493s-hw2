import torch

from model import GPT
from model import GPTConfig 
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy 

#run one step of inference on the model
def inference(model_weights, config, input_text=""):
    #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    transformer = GPT(config).to(device_type)
    transformer.load_state_dict(torch.load(model_weights, weights_only= True, map_location=device_type))

    #read data
    d_tokens = config.tokenizer(input_text).input_ids
    d_tokens = d_tokens[:-1]  # Remove the CLS token
    pad = [0] * (config.block_size - len(d_tokens))
    d_tokens = pad + d_tokens  # Pad the input tokens
    x_tokens = torch.tensor(d_tokens, dtype = torch.long)
    x_tokens = torch.tensor(x_tokens).unsqueeze(0)  # Add batch dimension

    #run inference
    pad_id = config.tokenizer.pad_token_id
    attention_mask = (x_tokens != pad_id).bool()
    logits = transformer(x_tokens, attention_mask=attention_mask)
    print(logits.shape)
    predicted_token = logits.argmax(dim=-1).squeeze()
    print(f"Predicted token: {predicted_token}")
    predicted_text = config.tokenizer.decode(predicted_token)
    print(f"Input: {config.tokenizer.decode(d_tokens)}")    
    print(f"Predicted: {predicted_text}")

def infer(model, config, input=""):
        #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    transformer = model.to(device_type)

    #read data
    d_tokens = config.tokenizer(input).input_ids
    d_tokens = d_tokens[:-1]  # Remove the CLS token
    pad = [0] * (config.block_size - len(d_tokens))
    d_tokens = pad + d_tokens  # Pad the input tokens
    x_tokens = torch.tensor(d_tokens, dtype = torch.long)
    x_tokens = torch.tensor(x_tokens).unsqueeze(0)  # Add batch dimension

    #run inference
    pad_id = config.tokenizer.pad_token_id
    attention_mask = (x_tokens != pad_id).bool()
    logits = transformer(x_tokens, attention_mask=attention_mask)
    logits = logits[:, -1, :]  # Get the logits for the last token
    predicted_token = logits.argmax(dim=-1).squeeze()
    
    print(f"Predicted token: {predicted_token}")
    predicted_text = config.tokenizer.decode(predicted_token)
    print(f"Input: {config.tokenizer.decode(d_tokens)}")    
    print(f"Predicted: {predicted_text}")
    return predicted_token, predicted_text
