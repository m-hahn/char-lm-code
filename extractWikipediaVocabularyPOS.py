from paths import WIKIPEDIA_HOME
unigrams = {}

if False:
 pathIn = WIKIPEDIA_HOME+"german-train-tagged.txt"
 pathOut = WIKIPEDIA_HOME+"german-wiki-word-vocab-POS.txt"
else:
 pathIn = "/checkpoint/mbaroni/char-rnn-exchange/tagged_italian_training_corpus.txt"
 pathOut = WIKIPEDIA_HOME+"itwiki/italian-wiki-word-vocab-POS.txt"

import random
with open(pathIn, "r") as inFile:
   for line in inFile:
      line = line[:-1]
      index = line.find("\t")
      if index == -1:
#         print(line)
         continue
#      word = line[:index].lower()
      index2 = line.find("\t", index+1)
      if index2 == -1:
          continue
 #     pos = line[index+1:index2]
      word = line[:index2].lower()
      unigrams[word] = unigrams.get(word, 0) + 1
 #     if random.random() > 0.99:
#          break
unigrams = sorted(list(unigrams.items()), key=lambda x:x[1],reverse=True)
with open(pathOut, "w") as outFile:
  for word, count in unigrams:
      print(f"{word}\t{count}", file=outFile)
      

