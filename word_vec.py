import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklean

USE_CUDA = torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

C = 3  # windows
K = 100  # negative sample
NUM_EPOCH = 2
MAX_VOCAB_SIZE = 120
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100


def word_tokenize(text):
    return text.split()


with open("text8.train.txt", 'r')as fin:
    text = fin.read()

text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab['<unk>'] = len(text)-np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx ={word: i for i ,word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
word_freqs=word_freqs **(3./4.)#float
word_freqs=word_counts/np.sum(word_counts)


