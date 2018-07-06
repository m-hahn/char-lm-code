
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
#parser.add_argument("--sequence_length", type=int, default=10)


args=parser.parse_args()
print(args)





from corpusIterator import CorpusIterator
training = CorpusIterator(args.language, partition="train", storeMorph=False, removePunctuation=True)
dev = CorpusIterator(args.language, partition="dev", storeMorph=False, removePunctuation=True)

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open("/checkpoint/mhahn/char-vocab-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary
    for sentence in plus(training.iterator(), dev.iterator()):
       for line in sentence:
           for char in line["word"]:
              char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open("/checkpoint/mhahn/char-vocab-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn_encoder = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=True).cuda()
rnn_decoder = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=True).cuda()


rnn_parameter_names = [name for name, _ in rnn_encoder.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_encoder_drop = WeightDrop(rnn_encoder, [(name, args.weight_dropout_in) for name, _ in rnn_encoder.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn_encoder.named_parameters() if name.startswith("weight_hh_")])
rnn_decoder_drop = WeightDrop(rnn_decoder, [(name, args.weight_dropout_in) for name, _ in rnn_decoder.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn_decoder.named_parameters() if name.startswith("weight_hh_")])


output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=3)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn_encoder, rnn_decoder, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn_encoder" : rnn_encoder, "rnn_decoder" : rnn_decoder, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load("/checkpoint/mhahn/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


import numpy.random

def prepareDataset(data, train=True):
      data = list(data)
      random.shuffle(data)
      boundaries = [[]]
      numeric = [[0,0]]
      currentLength = int(numpy.random.gamma(shape= 15.0, scale=1.0))
      for sentence in data:
         for word in sentence:
             for char in word["word"]:
                numeric[-1].append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
                if len(numeric[-1]) > currentLength:
#                   assert len(numeric[-1]) == currentLength+1, currentLength
                   numeric[-1].append(0)
                   numeric[-1].append(0)
                   numeric.append([0,0])
                   boundaries.append([])
                   currentLength = int(numpy.random.gamma(shape= 15.0, scale=1.0))
             boundaries[-1].append(len(numeric[-1]))
         numeric[-1].append(1)
         if len(numeric[-1]) > currentLength:
#            assert len(numeric[-1]) == currentLength+1
            numeric[-1].append(0)
            numeric[-1].append(0)
            numeric.append([0,0])
            boundaries.append([])
            currentLength = int(numpy.random.gamma(shape= 15.0, scale=1.0))
         boundaries[-1].append(len(numeric[-1]))
      numeric[-1] = numeric[-1] + ([0]*(currentLength + 3 - len(numeric[-1])))
 #     assert len(numeric[-1]) == currentLength + 3
      result= list(zip(numeric, boundaries))
      result = sorted(result, key=lambda x:len(x[0]))
      bins = [result[offset:offset+args.batchSize] for offset in range(0, len(result), args.batchSize)]
      random.shuffle(bins)
      result = []
      for data_bin in bins:
        for data in data_bin:
            result.append(data)
      return result

def forward(numericAndBoundaries, train=True, printHere=False):
  #    print("hgbghjjh")
 #     print(numericAndBoundaries)
      numeric, boundaries = zip(*numericAndBoundaries)
#      print(numeric)
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:-1].cuda(), requires_grad=False)
      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(len(numeric[0])-2, len(numeric), 1, 1)
      target_tensor_backward = Variable(torch.LongTensor(numeric).transpose(0,1)[:-2].cuda(), requires_grad=False).view(len(numeric[0])-2, len(numeric), 1, 1)
      target_tensor = torch.cat([target_tensor_forward, target_tensor_backward], dim=2)

    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      _, encoded = rnn_encoder_drop(embedded, None)

      out, _ = rnn_decoder_drop(embedded, encoded)
      out = out.view(len(numeric[0])-2, len(numeric), 2, -1)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(len(numeric[0])-2, len(numeric), 2)
         losses = lossTensor.data.cpu().numpy()
         boundaries_index = [0 for _ in numeric]
         for i in range((len(numeric[0])-2)-1):
            if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
               boundary = True
               boundaries_index[0] += 1
            else:
               boundary = False
            print((losses[i][0][0], losses[i][0][1], itos[numeric[0][i+1]-3], boundary))
      return loss, len(numeric) * (len(numeric[0])-2)

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      optim.step()




def padWords(numeric):
   maxLength = max([len(x[0]) for x in numeric])
   for i, words in enumerate(numeric):
       numeric[i] = (words[0] + ([0]*(maxLength - len(words[0]))), words[1])
   return numeric 

import time

devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = training.iterator()
   numericAndBoundaries = prepareDataset(training_data, train=True) # redo for each epoch




#   oldParameters = {}
#   for name, param in rnn.named_parameters():
#      oldParameters[name] = param
#      setattr(rnn, name, torch.nn.functional.dropout(param, p=weight_dropout))
#      print(name, param.size())
#

   rnn_encoder_drop.train(True)
   rnn_decoder_drop.train(True)
   startTime = time.time()
   trainChars = 0
   for offset in range(0, len(numericAndBoundaries), args.batchSize):
      printHere = (int(offset/args.batchSize) % 50 == 0)
      loss, charCounts = forward(padWords(numericAndBoundaries[offset:offset+args.batchSize]), printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts 
      if printHere:
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

   rnn_encoder_drop.train(False)
   rnn_decoder_drop.train(False)
#   for name, param in rnn.named_parameters():
#      setattr(rnn, name, oldParameters[name])


     
   dev_loss = 0
   dev_char_count = 0
   dev_data = dev.iterator()
   numericAndBoundaries = prepareDataset(dev_data, train=False)

   for offset in range(0, len(numericAndBoundaries), args.batchSize):
       printHere = (int(offset/args.batchSize) % 10 == 0)
       loss, numberOfCharacters = forward(padWords(numericAndBoundaries[offset:offset+args.batchSize]), printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()[0]
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
   with open("/checkpoint/mhahn/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
      print(" ".join([str(x) for x in devLosses]), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), "/checkpoint/mhahn/"+args.save_to+".pth.tar")


