from model import GPT
from model import GPTConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

def train(data, weight_decay, learning_rate, betas):
    
    #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    config = GPTConfig()
    transformer = GPT(config).to(device_type)
    optimizer = transformer.configure_optimizers(weight_decay, learning_rate, betas, device_type)

    #read data  
    train_x = []
    train_y = []
    with open(data, "r") as f:
        d = f.readline()
        while d != "":
            
            d_tokens = config.tokenizer(d).input_ids
            length = len(d_tokens)
            #pad = [config.pad_id] * (config.block_size - length)
            #d_tokens = pad + d_tokens
            if length < 3:
                d = f.readline()
                continue
            '''''
            x = d_tokens[:-1]
            y = d_tokens[1:]
            print(f"X: {x} | Y: {y}")
            x_tokens = torch.tensor(x, dtype = torch.long)
            y_tokens = torch.tensor(y, dtype = torch.long)
            train_x.append(x_tokens)
            train_y.append(y_tokens)
            '''
            
            for i in range(length):
                x = d_tokens[:-1]
                x = x[-config.block_size:]
                y = d_tokens[1:]
                y = y[-config.block_size:]

                print(f"X: {x} | Y: {y}")
                x_tokens = torch.tensor(x, dtype = torch.long)
                y_tokens = torch.tensor(y, dtype = torch.long)
                train_x.append(x_tokens)
                train_y.append(y_tokens)
                d_tokens = d_tokens[:-1] 
            
            d = f.readline()

    # pad sequences to max length
    pad_id = config.tokenizer.pad_token_id
    train_x = pad_sequence(train_x, batch_first=True, padding_value=pad_id)
    train_y = pad_sequence(train_y, batch_first=True, padding_value=-100)



    train_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_set, batch_size = 32, shuffle= True)

   
    #loop batches of data
    transformer.train()
    end = False
    for epoch in range(500):
        print(f"Epoch {epoch + 1}")
        for idx, (X, y) in enumerate(train_loader):
            X = X.to(device_type)
            y = y.to(device_type)
            #print(f"Batch {idx} | X shape: {X.shape} | y shape: {y.shape}")
            #print(f"Batch {idx} | X: {X} | y: {y}")

            attention_mask = (X != pad_id).bool()  # Create attention mask where pad_id is 0 and others are 1
            #print(attention_mask)

        #run forward pass on batch
            logits = transformer(X, self_attention_mask=attention_mask)
        #compute loss
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index= -100)
        #call optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if idx % 10 == 0:
                print(f"Loss: {loss.item():.4f}")
            if(loss.item() < 0.0001):
                print("Loss is low enough, stopping training")
                end = True
        if end:
            break
    #save model
    torch.save(transformer.state_dict(), "GPT_simple_data")
