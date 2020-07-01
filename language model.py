import torch
import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)
BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000
HIDDEN_SIZE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path='E:\\gre\\text8\\', train='text8.train.txt',
                                                                     validation='text8.dev.txt',
                                                                     test='text8.test.txt', text_field=TEXT)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
VOCAB_SIZE = len(TEXT.vocab)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=BATCH_SIZE,
                                                                     device=torch.device("cuda"),
                                                                     bptt_len=40, repeat=False, shuffle=True)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, text, hidden):
        emb = self.embed(text)
        output, hidden = self.lstm(emb, hidden)
        # print(output.shape)
        # output = output.view(-1, output.shape[2])
        # print(output.shape)
        out_vocab = self.decoder(output.view(-1, output.shape[2]))
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1))
        return out_vocab, hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        return weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True), \
               weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True)


model = RNNModel(vocab_size=len(TEXT.vocab), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


NUM_EPOCHS = 2
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
GRAD_CLIP = 5.0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)

        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("loss", loss.item())
        # if i % 10000:
        #     val_loss = evaluate(model,val_iter)
        #     torch.save(model.state_dict(),"lm.pth")
