
import torch
import random
import numpy as np
def generate_data(p, data_size, splits = (0.8, 0.1, 0.1)):
    train = "train"
    train_out = ""
    test = "test"
    test_out = ""
    val = "val"
    val_out = ""
    data = [train, test, val]
    for i in range(data_size):
        a = random.randint(0, p)
        b = random.randint(0, a)
        c = (a + b) % p 
        split = random.choices(data, splits)[0]
        if(split == "train"):
            train_out = train_out + str(a) + " + " + str(b) + " = " + str(c) + "\n"
        elif(split == "test"):  
            test_out = test_out + str(a) + " + " + str(b) + " = " + str(c) + "\n"
        elif(split == "val"):   
            val_out = val_out + str(a) + " + " + str(b) + " = " + str(c) + "\n"
        #print(split)
    
    for i in range(data_size):
        a = random.randint(0, p)
        b = random.randint(0, a)
        c = np.abs(a - b) % p
        split = random.choices(data, splits)[0]
        if(split == "train"):
            train_out = train_out + str(a) + " - " + str(b) + " = " + str(c) + "\n"
        elif(split == "test"):  
            test_out = test_out + str(a) + " - " + str(b) + " = " + str(c) + "\n"
        elif(split == "val"):   
            val_out = val_out + str(a) + " - " + str(b) + " = " + str(c) + "\n"


    for i in range(data_size):
        a = random.randint(0, p)
        b = random.randint(1, p-1) #b should not be 0 to avoid division by zero
        print(a, b)
        modular_inverse = pow(b, -1, p) #The pow(base, exp, mod) function can compute the modular inverse when the modulus is prime or when the numbers are coprime.
        c = (a * modular_inverse) % p
        split = random.choices(data, splits)[0]
        if(split == "train"):
            train_out = train_out + str(a) + " / " + str(b) + " = " + str(c) + "\n"
        elif(split == "test"):  
            test_out = test_out + str(a) + " / " + str(b) + " = " + str(c) + "\n"
        elif(split == "val"):   
            val_out = val_out + str(a) + " / " + str(b) + " = " + str(c) + "\n"
    
    with open("train.txt", "w") as f:
        f.write(train_out)
    with open("test.txt", "w") as f:
        f.write(test_out)   
    with open("val.txt", "w") as f:
        f.write(val_out)

generate_data(97, 50, [0.8, 0.1, 0.1])