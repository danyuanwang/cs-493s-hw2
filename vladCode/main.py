from model import *
from train import *
from transformers import GPT2TokenizerFast
import random
import numpy as np
import torch
from inference import *
def main():
    print("LETS GO")
    #test1p5()
    train(1000, 1, 0.001, [0.9, 0.98], "1Decay-0.1Dropout-divOnly-97P-2048Data-2Layers-1000Epochs-SEED=12-updatedDataGen", 97, True, 2, 2048, 256, 12)
#train(num_epochs, weight_decay, learning_rate, betas, label, pValue, allowDiv, numLayers, dataSize, batch_size, seed: int = 493):
   
def test1p5():
    #train(100, 1, 0.001, [0.9, 0.999], "sec1.5", -1, False, 1, 100, 256, 493)
    infer("sec1.5resultingModel.pth", "iloveml.txt", 1, 4, "ilovemlResults.txt")
def test2():
    train(100, 1, 0.001, [0.9, 0.999], "plusminus97P1024Data1Layer100Epochs", 97, False, 1, 1024, 256, 493)
    train(100, 1, 0.001, [0.9, 0.999], "plusminus113P1024Data1Layer100Epochs", 113, False, 1, 1024, 256, 493)
    train(100, 1, 0.001, [0.9, 0.999], "plusminus97P1024Data2Layers1000Epochs", 97, False, 2, 1024, 256, 493)
    train(1000, 1, 0.001, [0.9, 0.98], "1Decay-0.1Dropout-divOnly-97P-2048Data-2Layers-1000Epochs-SEED=12-updatedDataGen", 97, True, 2, 2048, 256, 12)
if __name__ == "__main__":
    main()