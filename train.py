from model import GPT
from model import GPTConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from test_loss import get_test_loss
import matplotlib.pyplot as plt

def plot(test_losses, train_losses, iters):
    plt.figure(figsize=(10, 6))
    plt.plot(iters, train_losses, label='Train Loss')
    plt.plot(iters, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
def train(data, weight_decay, learning_rate, betas, model = None):
    
    #device management
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    #instantiate model
    if model is None:
        config = GPTConfig()
        config.vocab_size = 50304
        config.block_size = 10
        config.n_layer = 1
        transformer = GPT(config).to(device_type)
        optimizer = transformer.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    else:
        config = GPTConfig()
        transformer = model.to(device_type)
        optimizer = transformer.configure_optimizers(weight_decay, learning_rate, betas, device_type)

    #read data  
    train_x = []
    train_y = []
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
            
            print(d_tokens)
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
            '''
            
            d = f.readline()

    # pad sequences to max length
    pad_id = config.tokenizer.pad_token_id
    train_x = pad_sequence(train_x, batch_first=True, padding_value=pad_id)
    train_y = pad_sequence(train_y, batch_first=True, padding_value=-100)



    train_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_set, batch_size = 100, shuffle= True)

   
    #loop batches of data
    transformer.train()
    end = False
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1000):
        print(f"Epoch {epoch + 1}")
        train_loss = 0
        total = 0
        correct = 0
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
            train_loss += loss.item()
        #compute accuracy

            non_pad = y != -100
            preds = logits.argmax(dim=-1)
            correct += (preds[non_pad] == y[non_pad]).sum().item()
            total += non_pad.sum().item()
            #print(preds[non_pad])
            #print(y[non_pad])
        #call optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if idx % 10 == 0:
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {100 * correct / total:.4f}%")
            if(loss.item() < 0.0001):
                print("Loss is low enough, stopping training")
                end = True
                break
        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        test_loss, test_accuracy = get_test_loss( "test.txt", transformer, config)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
        if end:
            break

    #save model
    #print(train_losses)
    #print(test_losses)
    plot(test_losses, train_losses, range(1000))
    plot(test_accuracies, train_accuracies, range(1000))
    torch.save(transformer.state_dict(), "number_model_4_grok.pth")
