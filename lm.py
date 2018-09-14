
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
args=parser.parse_args()
print(args)

with open("/private/home/mhahn/data/UD_English-EWT/en_ewt-ud-train.conllu", "r") as inFile:
    data = [x.split("\t")[1].lower() for x in inFile.read().split("\n") if len(x) > 1 and not x.startswith("#")]
    #print(data)
    data = "".join(data)
    #print(data)
itos = list(set([x for x in data]))
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])


sequenceLength = 100
batchSize = 16

partitions = [sequenceLength*x for x in range(int(len(data)/sequenceLength))]



import random


random.Random(4377).shuffle(partitions)


train = partitions[100:]
dev = partitions[:100]

import torch

print(torch.__version__)

rnn = torch.nn.LSTM(100, 1024).cuda()

output = torch.nn.Linear(1024, len(itos)+1).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+1, embedding_dim=100).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss()
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False)
char_dropout = torch.nn.Dropout2d(p=0.33)
#dropout = torch.nn.Dropout(p=0.33)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=0.1, momentum=0.0)

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load("/checkpoint/mhahn/"+__file__+"_"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable



def forward(indices, train=True, printHere=False):
      numeric = [([0] + [stoi[data[x]]+1 for x in range(b, b+sequenceLength) if x < len(data)]) for b in indices]
     # print(numeric)
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)


    #  print(char_embeddings)
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, _ = rnn(embedded, None)
      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)
      if printHere:
         loss = print_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1)).view(sequenceLength, batchSize)
         losses = loss.data.cpu().numpy()
         for i in range(sequenceLength-1):
            print((losses[i][0], itos[numeric[0][i+1]-1]))
         loss = loss.mean()
      else:     
         loss = train_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1))
      return loss

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      optim.step()
      



for epoch in range(100):
   print(epoch)
   random.shuffle(train)
   for batch in range(int(len(train)/batchSize)):
      printHere = (batch % 50 == 0)
      loss = forward(train[batch*batchSize:(batch+1)*batchSize], printHere=printHere)
      backward(loss, printHere)
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), "/checkpoint/mhahn/"+__file__+"_"+args.save_to+".pth.tar")


##      print(train[batch*batchSize:(batch+1)*batchSize])
#      numeric = [([0] + [stoi[data[x]]+1 for x in range(b, b+sequenceLength) if x < len(data)]) for b in train[batch*batchSize:(batch+1)*batchSize]]
#     # print(numeric)
#      input_tensor = Variable(torch.LongTensor(numeric[:-1]).transpose(0,1).cuda(), requires_grad=False)
#      target_tensor = Variable(torch.LongTensor(numeric[1:]).transpose(0,1).cuda(), requires_grad=False)
#
#    #  print(char_embeddings)
#      embedded = char_embeddings(input_tensor)
#      out, _ = rnn(embedded, None)
#      logits = output(out) 
#      log_probs = logsoftmax(logits)
#   #   print(logits)
#  #    print(log_probs)
# #     print(target_tensor)
#      loss = train_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1))
#      optim.zero_grad()
#      if batch % 10 == 0:
#         print(loss)
#      loss.backward()
#      optim.step()
#      
#
