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

# import sklearn

USE_CUDA = 1
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
torch.cuda.manual_seed(53113)

C = 3  # windows
K = 100  # negative sample
NUM_EPOCH = 2
MAX_VOCAB_SIZE = 30000
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100
BATCH_SIZE = 128  # the batch size


def word_tokenize(text):
    return text.split()


with open("E:\\gre\\text8\\text8.train.txt", 'r')as fin:
    text = fin.read()

text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
# print(type(word_to_idx))

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)  # float
word_freqs = word_counts / np.sum(word_counts)
VOCAB_SIZE = len(idx_to_word)
print(VOCAB_SIZE)


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, VOCAB_SIZE - 1) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts =torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        # fucaiyang
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


class EmbeddingModel(nn.Module):
    def __init__(self, vcab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vcab_size
        self.embed_size = embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        #self.out_embed.weight.data.uniform_(-initrange, initrange)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        #self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.out_embed(pos_labels)
        neg_embedding = self.out_embed(neg_labels)

        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze()
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze()
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg
        return -loss

    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCH):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        # print(input_labels,pos_labels,neg_labels)
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch", e, "iteration", i, loss.item())
