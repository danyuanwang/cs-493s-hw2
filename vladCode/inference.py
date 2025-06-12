import torch
from model import *

def infer(state, text, response_length, block_size, output_file):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig()
    config.block_size = block_size #possibly redundant 
    model = GPT(config).to(dev)
    tokenizer = config.tokenizer
    model.load_state_dict(torch.load(state, map_location=dev))
    model.eval()

    with open(text, 'r') as f:
        lines = f.readlines()
    
    outs = []

    #feed each line into the model talker 2000
    for line in lines:
        with torch.no_grad():
            out = talk(model, line, tokenizer,  block_size, response_length, dev)
            outs.append(out)
            print(out)
    with open(output_file, 'w') as f:
        for out in outs:
            f.write(out + '\n')
    return outs

def talk(model, line, tokenizer, block_size, response_length, dev):
    model.eval()
    with torch.no_grad(): 
        tokens = tokenizer.encode(line)
        out = tokenizer.encode(line)
        for i in range(response_length):
            tokensTensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(dev) 
            logits = model(tokensTensor)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).item() 
            tokens = tokens + [next_token]
            out = out + [next_token]
            if len(tokens) > block_size:
                tokens = tokens[1:]
    return tokenizer.decode(out)
