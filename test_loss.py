from model import GPT
from model import GPTConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from inference import infer

def get_test_loss(data, model, config):
    
    #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    transformer = model.to(device_type)

    #read data  
    X = []
    Y = []
    with open(data, "r") as f:
        d = f.readline()
        while d != "":
            d = d.replace('\n', '')
            d_tokens = config.tokenizer(d).input_ids
            length = len(d_tokens)
            #pad = [config.pad_id] * (config.block_size - length)
            #d_tokens = pad + d_tokens
            if length < 3:
                d = f.readline()
                continue
            
            x = d_tokens[:-1]
            y = d_tokens[1:]
            #print(f"X: {x} | Y: {y}")
            x_tokens = torch.tensor(x, dtype = torch.long)
            y_tokens = torch.tensor(y, dtype = torch.long)
            X.append(x_tokens)
            Y.append(y_tokens)
            
            '''
            for i in range(length):
                x = d_tokens[:-1]
                x = x[-config.block_size:]
                y = d_tokens[1:]
                y = y[-config.block_size:]

                #print(f"X: {x} | Y: {y}")
                x_tokens = torch.tensor(x, dtype = torch.long)
                y_tokens = torch.tensor(y, dtype = torch.long)
                X.append(x_tokens)
                Y.append(y_tokens)
                d_tokens = d_tokens[:-1] 
            '''
            d = f.readline()

    # pad sequences to max length
    pad_id = config.tokenizer.pad_token_id
    X = pad_sequence(X, batch_first=True, padding_value=pad_id)
    Y = pad_sequence(Y, batch_first=True, padding_value=-100)



    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle= True)

   
    #loop batches of data
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():

        for idx, (X, y) in enumerate(dataloader):
            X = X.to(device_type)
            y = y.to(device_type)
            #print(f"Batch {idx} | X shape: {X.shape} | y shape: {y.shape}")
            #print(f"Batch {idx} | X: {X} | y: {y}")

            attention_mask = (X != pad_id).bool()  # Create attention mask where pad_id is 0 and others are 1
            #print(attention_mask)

            #run forward pass on batch
            logits = transformer(X, self_attention_mask=attention_mask)
            #compute loss
            loss += cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index= -100)
            non_pad = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[non_pad] == y[non_pad]).sum().item()
            total += non_pad.sum().item()

            #if idx % 10 == 0:
            #    print(f"Loss: {loss.item():.4f}")
    return loss.item() / len(dataloader), 100 * correct/total  # Average loss over all batches
    
if __name__ == "__main__":
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig()
    transformer = GPT(config).to(device_type)
    transformer.load_state_dict(torch.load("number_model_1_layer_1.pth", weights_only= True, map_location=device_type))
    loss = get_test_loss("test.txt", transformer, config)