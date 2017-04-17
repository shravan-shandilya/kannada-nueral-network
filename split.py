import numpy as np
import random

split_percentage = 75
total_samples_per_alphabet = 25
train_set_samples = (split_percentage*total_samples_per_alphabet) / 100
test_set_samples = (100-split_percentage)*total_samples_per_alphabet /100

dataset = np.genfromtxt("kannada_grayscale_temp.csv",delimiter=",")
train_set_sample_index_positives = random.sample(range(1,25),train_set_samples)
test_set_sample_index_positives = random.sample(range(1,25),test_set_samples)

train_set_sample_index_negetives = random.sample(range(26,400),train_set_samples)
test_set_sample_index_negetives = random.sample(range(26,400),test_set_samples)

train_set = np.zeros((train_set_samples*2,401))
test_set = np.zeros((test_set_samples*2,401))

i = 0
for index in train_set_sample_index_positives:
    train_set[i] = dataset[index][:401]
    i=i+1

for index in train_set_sample_index_negetives:
    train_set[i] = dataset[index][:401]
    i=i+1
i = 0
for index in test_set_sample_index_positives:
    test_set[i] = dataset[index][:401]
    i=i+1

for index in test_set_sample_index_negetives:
    test_set[i] = dataset[index][:401]
    i=i+1

np.savetxt("train_set.csv",train_set,delimiter=",",fmt="%5.2f")
np.savetxt("test_set.csv",test_set,delimiter=",",fmt="%5.2f")
