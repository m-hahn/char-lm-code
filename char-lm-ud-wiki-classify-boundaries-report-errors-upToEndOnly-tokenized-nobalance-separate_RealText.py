# Enforces balancing in the training set, and enforces disjointness of training and test sets
# (This should be used for evaluation)


# python char-lm-ud-wiki-classify-boundaries-report-errors-upToEndOnly-tokenized-nobalance-separate_RealText.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

# python char-lm-ud-wiki-classify-boundaries-report-errors-upToEndOnly-tokenized-nobalance-separate_RealText.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

# python char-lm-ud-wiki-classify-boundaries-report-errors-upToEndOnly-tokenized-nobalance-separate_RealText.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412



TRAINING_SIZE = 1000

from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)

import random
random.seed(1) # May nonetheless not be reproducible, since the classifier library doesn't seem to allow setting the seed

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
parser.add_argument("--sequence_length", type=int, default=50)


args=parser.parse_args()
print(args)


#assert args.language == "german"


import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary

    with open(WIKIPEDIA_HOME+"/"+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      boundariesAll = [None for _ in range(args.sequence_length+1)]

      count = 0
      currentWord = ""
      print("Prepare chunks")
      for chunk in data:
          print(len(chunk))
          for word in chunk:
#             assert word != "popula"
             for char in word:
                if boundariesAll[len(numeric)] is None:
                      boundariesAll[len(numeric)] = currentWord
       
                count += 1
                currentWord += char
                numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
                if len(numeric) > args.sequence_length:
                   yield numeric, boundaries, boundariesAll
                   numeric = [0]
                   boundaries = [None for _ in range(args.sequence_length+1)]
                   boundariesAll = [None for _ in range(args.sequence_length+1)]
      
             assert currentWord == word, (currentWord, word)
             boundaries[len(numeric)] = currentWord
             boundariesAll[len(numeric)] = currentWord
       
             currentWord = ""
       
  

# from each bath element, get one positive example OR one negative example

wordsSoFar = set()
hidden_states = []
labels = []
relevantWords = []
relevantNextWords = []
labels_sum = 0

boundary_positions = []

def forward(numeric, train=True, printHere=False, enforceBalancing=True):
      global labels_sum
      numeric, boundaries, boundariesAll = zip(*numeric)

      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)

      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, _ = rnn_drop(embedded, None)

      for i in range(len(boundaries)): # for each batch sample
         target = (labels_sum + 10 < len(labels)/2) or (random.random() < 0.5) # decide whether to get positive or negative sample
         true = sum([((x == None) if target == False else (x is not None and y not in wordsSoFar)) for x, y in list(zip(boundaries[i], boundariesAll[i]))[int(args.sequence_length/2):-10]]) # condidates
 #        print(train, target, true)
         if train and true == 0:
            continue
         soFar = 0
         for j in range(int(len(boundaries[i])/2), len(boundaries[i])-10):
 #          if (not train) or (lambda x, y:((x is None if target == False else (x is not None and y not in wordsSoFar))))(boundaries[i][j], boundariesAll[i][j]):
 #             if (not train) or random.random() < 1/(true-soFar):
                  if boundaries[i][j] is None:
                     continue
#                  assert (boundaries[i][j] is not None) == target, (boundaries[i][j], boundariesAll[i][j], target)
  #                print("ADDING", target)
                  hidden_states.append(out[j-1,i].detach().data.cpu().numpy())
                  boundary_positions.append((i,j))

                  labels.append(1 if target else 0)
                  relevantWords.append(boundariesAll[i][j])
                  relevantNextWords.append(([boundaries[i][k] for k in range(j+1, len(boundaries[i])) if boundaries[i][k] is not None]+["END_OF_SEQUENCE"])[0])
#                  print(target, relevantWords[-1], relevantNextWords[-1])
                  assert boundariesAll[i][j] is not None

                  labels_sum += labels[-1]
                  if enforceBalancing:
                     wordsSoFar.add(boundariesAll[i][j])
                  break
              #soFar += 1
 #        assert soFar < true





import time

devLosses = []
#for epoch in range(10000):
if True:
   training_data = corpusIteratorWikiWords.dev(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_drop.train(False)
   startTime = time.time()
   trainChars = 0
   counter = 0
   while True:
      counter += 1
      try:
         numeric = [next(training_chars) for _ in range(args.batchSize)]
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      forward(numeric, printHere=printHere, train=True)
      #backward(loss, printHere)
      if printHere:
          print((counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

      if len(labels) > TRAINING_SIZE:
         break
  

predictors = hidden_states
dependent = labels


TEST_FRACTION = 0.9
print("Creating regression model")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, words_train, words_test, next_words_train, next_words_test = train_test_split(predictors, dependent, relevantWords, relevantNextWords, test_size=TEST_FRACTION, random_state=random.randint(1,100), shuffle=True)


from sklearn.linear_model import LogisticRegression

print("regression")

logisticRegr = LogisticRegression()

print(y_train)

logisticRegr.fit(x_train, y_train)


errors = []
scores = []

examples_count = 0

correct = 0
falsePositives = 0
falseNegatives = 0

#for batch in range(1): # Go through the entire remainder of the Dev set
if True:
     
     devLosses = []
     #for epoch in range(10000):
     if True:
     

        rnn_drop.train(False)
        startTime = time.time()
        trainChars = 0
        counter = 0
        for _ in range(1):
           hidden_states = []
           labels = []
           relevantWords = []
           relevantNextWords = []
           labels_sum = 0
           boundary_positions = []

           counter += 1
           try:
              numeric = [next(training_chars) for _ in range(args.batchSize)]
           except StopIteration:
              break
           printHere = (counter % 50 == 0)
           forward(numeric, printHere=printHere, train=False, enforceBalancing=False)
           #backward(loss, printHere)
           if printHere:
               print((counter))
               print("Dev losses")
               print(devLosses)
               print("Chars per sec "+str(trainChars/(time.time()-startTime)))
     
           #if len(labels) > 10000:
          #    break
           if len(hidden_states) == 0:
                break
           predictors = hidden_states
           dependent = labels
           
           x_test = predictors
           y_test = dependent
           words_test = relevantWords
           next_words_test = relevantNextWords
           
           
           
           
           predictions = logisticRegr.predict(x_test)
           
           
           score = logisticRegr.score(x_test, y_test)
           scores.append(score)
      
           parts = [[] for x in range(args.batchSize)]
           
      
           for i in range(len(x_test)):
               parts[boundary_positions[i][0]].append((boundary_positions[i][1], y_test[i], predictions[i], words_test[i], next_words_test[i]))
           for p in parts:
              p = sorted(p, key=lambda x:x[0])
              print(p)
              pred_string = "".join([c[3][-1]+(" " if c[2] == 1 else "") for c in p])
              real_string = "".join([c[3][-1]+(" " if c[1] == 1 else "") for c in p])

              print(real_string)
              print(pred_string)
      #         print(i, y_test[i], predictions[i], words_test[i], next_words_test[i], boundary_positions[i])
      
      
           for i in range(len(predictions)):
               if predictions[i] != y_test[i]:
                     errors.append((y_test[i], (words_test[i], next_words_test[i], predictions[i], y_test[i])))
                     if predictions[i] == 1:
                       falsePositives += 1
                     elif predictions[i] == 0:
                       falseNegatives += 1
               else:
                  correct += 1
           print("Balance ",sum(y_test)/len(y_test))
           examples_count += len(y_test)
 #       if len(hidden_states) == 0:
#                break


#correct = 0
#falsePositives = 0
#falseNegatives = 0
precision = correct / (correct + falsePositives)
recall = correct / (correct + falseNegatives)

print("Boundary measures", "Precision", precision, "Recall", recall, "F1", 2*(precision*recall)/(precision+recall))


print(examples_count)
score = sum(scores)/len(scores)
print(score)

