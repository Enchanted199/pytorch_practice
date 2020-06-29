import torch
y = torch.empty(5,3)
result = torch.empty(5,3)
print(result)
print(y)

#+=操作
y.add_(result)
print(y)
print(y[1:,1:])
x= y.view(15)
print(x)#only one row
x=y.view(3,-1) #-1 will be computed by torch

#x.item()#get data
if torch.cuda.is_available():
    print(1231231321313)
else:
    print(torch.version.cuda)


#lec02
print(torch.cuda.get_device_name(0))