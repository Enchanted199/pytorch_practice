import numpy as np
import torch

NUM_DIGITS = 10

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
trX=trX.cuda()
trY=trY.cuda()
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)
if torch.cuda.is_available():
    model= model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(),lr=0.05)

BATCH_SIZE=128

for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end = start+BATCH_SIZE
        batchX=trX[start:end]
        batchY=trY[start:end]
        #print(batchY.is_cuda)
        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)
        print("Epoch",epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

testX=torch.Tensor({binary_encode(i,NUM_DIGITS) for i in range(1,101)})
testX = testX.cuda()
with torch.no_grad():
    testY= model(testX)
predicts = zip(range(1,101), list(testY.max(1)[1]))
