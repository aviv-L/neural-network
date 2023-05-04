import numpy as np
import data_loader as dl
import train as train
import test as test
import json


print("Would you like to train or to test?")
choice = input("press 1 for training, 2 for testing.")

if choice == "1":
    train.train()
elif choice == "2":
    test.test()
else:
    print("invalid choice. try again.")

