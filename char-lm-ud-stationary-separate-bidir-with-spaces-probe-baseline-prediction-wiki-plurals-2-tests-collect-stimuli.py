
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

#parser.add_argument("--save-to", dest="save_to", type=str)

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
parser.add_argument("--sequence_length", type=int, default=50)


args=parser.parse_args()
print(args)





def plusL(its):
  for it in its:
       for x in it:
           yield x

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


import random




from corpusIterator import CorpusIterator


plurals = set()

training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True)

for sentence in training.iterator():
 for line in sentence:
   if line["posUni"] == "NOUN":
      morph = line["morph"]
      if "Number=Plur" in  morph and "Case=Dat" not in morph:
        if "|" not in line["lemma"] and line["lemma"].lower() != line["word"]:
          plurals.add((line["lemma"].lower(), line["word"]))

formations = {"e" : set(), "n" : set(), "s" : set(), "same" : set(), "r" : set()}

for singular, plural in plurals:
  if len(singular) == len(plural):
    if singular[-1] == plural[-1]:
      formations["same"].add((singular, plural))
    else:
       print((singular, plural))
  elif plural.endswith("n"):
     formations["n"].add((singular, plural))
  elif plural.endswith("s"):
     formations["s"].add((singular, plural))
  elif plural.endswith("e"):
     formations["e"].add((singular, plural))
  elif plural.endswith("r"):
     formations["r"].add((singular, plural))
  else:
      print((singular, plural))

#print(formations["n"])
#print(formations["same"])


print({x:len(y) for x, y in formations.items()})
quit()

def doChoiceList(xs):
    for x in xs:
       print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    print(losses)
    return np.argmin(losses)


def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x), encodeWord(y))
    print(losses)
    return 0 if losses[0] < losses[1] else 1


# classify singulars vs plurals
print("trained on n, s, e")

    

forNSE = list(plusL([formations["n"], formations["s"], formations["e"]]))

lengthsS = [0 for _ in range(55)]
lengthsP = [0 for _ in range(55)]

for sing, plur in forNSE:
   lengthsS[len(sing)] += 1
   lengthsP[len(plur)] += 1
   

lengths = [min(x,y) for x,y in zip(lengthsS, lengthsP)]

sumLengthsS = sum(lengthsS)
lengthsS = [float(x)/sumLengthsS for x in lengthsS]

sumLengthsP = sum(lengthsP)
lengthsP = [float(x)/sumLengthsP for x in lengthsP]

sumLengths = sum(lengths)
lengths = [float(x)/sumLengths for x in lengths]

ratioP = max([x/y if y > 0 else 0.0 for (x,y) in zip(lengths, lengthsP)])
ratioS = max([x/y if y > 0 else 0.0 for (x,y) in zip(lengths, lengthsS)])

import random

# from each type, sample N singulars and N plurals
N = 30
evaluationPoints = []

formationsBackup = formations

for _ in range(200):
     formations = {x : set(list(y)[:]) for x, y in formationsBackup.items()}


     singulars = {}
     plurals = {}
     for typ in ["n", "s", "e"]:
        singulars[typ] = []
        plurals[typ] = []
     
        formations[typ] = list(formations[typ])
        for _ in range(N):
           while True:
              index, sampledS = random.choice(list(zip(range(len(formations[typ])), formations[typ])))
              sampledS = sampledS[0]
              ratio = lengths[len(sampledS)] / (ratioS * lengthsS[len(sampledS)])
              assert 0<= ratio
              assert ratio <= 1
              if random.random() < ratio:
                  del formations[typ][index]
                  singulars[typ].append(sampledS)
                  break
              
           while True:
              index, sampledP = random.choice(list(zip(range(len(formations[typ])), formations[typ])))
              sampledP = sampledP[1]
              ratio = lengths[len(sampledP)] / (ratioP * lengthsP[len(sampledP)])
              assert 0<= ratio
              assert ratio <= 1
              if random.random() < ratio:
                 del formations[typ][index]
                 plurals[typ].append(sampledP)
                 break
     
     stratify_types = ["n" for _ in plurals["n"]] + ["s" for _ in plurals["s"]] + ["e" for _ in plurals["e"]]
     
     plurals = plurals["n"] + plurals["s"] + plurals["e"]
     singulars = singulars["n"] + singulars["s"] + singulars["e"]
     
     assert len(plurals) == len(singulars)
     
     
     print(singulars)
     print(plurals)
     print(len(plurals)) 
     print(sum([len(x) for x in plurals])/float(len(plurals)))
     print(sum([len(x) for x in singulars])/float(len(singulars)))
     
     
     encodedPlurals = encodeListOfWords(["."+y for y in plurals])
     encodedSingulars = encodeListOfWords(["."+x for x in singulars])
     
     #predictors = encodedSingulars + encodedPlurals
     
     #dependent = [0 for _ in encodedSingulars] + [1 for _ in encodedPlurals]
     
     from sklearn.model_selection import train_test_split
     sx_train, sx_test, sy_train, sy_test, st_train, st_test = train_test_split(encodedSingulars, [0 for _ in encodedSingulars], stratify_types, test_size=0.5, random_state=random.randint(0,100), shuffle=True, stratify = stratify_types)
     px_train, px_test, py_train, py_test, pt_train, pt_test = train_test_split(encodedPlurals, [1 for _ in encodedPlurals], stratify_types, test_size=0.5, random_state=random.randint(0,100), shuffle=True, stratify = stratify_types)
     
     x_train = sx_train + px_train
     x_test = sx_test + px_test
     y_train = sy_train + py_train
     y_test = sy_test + py_test
     t_train = st_train + pt_train
     t_test = st_test + pt_test
     
     
     print(y_train)
     print(y_test)
     
     
     from sklearn.linear_model import LogisticRegression
     
     print("regression")
     
     logisticRegr = LogisticRegression()
     
     logisticRegr.fit(x_train, y_train)
     
     predictions = logisticRegr.predict(x_test)
     
     for typ in ["n", "s", "e"]:
      indicesForType = [i for i in range(len(t_test)) if t_test[i] == typ]
      
      print(len(indicesForType))
      score = logisticRegr.score([x_test[i] for i in indicesForType], [y_test[i] for i in indicesForType])
      print(["test on "+typ,score])
     
      evaluationPoints.append((typ, score))
     
     #print(formations["r"])
     # test on R plurals
     encodedPluralsR = encodeListOfWords(["."+y for x, y in formations["r"]])
     encodedSingularsR = encodeListOfWords(["."+x for x, y in formations["r"]])
     
     predictors =  encodedSingularsR + encodedPluralsR
     dependent = [0 for _ in encodedSingularsR] + [1 for _ in encodedPluralsR]
     
     score = logisticRegr.score(predictors, dependent)
     print(["r plurals",score])
     
     evaluationPoints.append(("R", score))
     
     # test on R plurals
     encodedPluralsR = encodeListOfWords(["."+y for x, y in formations["same"]])
     encodedSingularsR = encodeListOfWords(["."+x for x, y in formations["same"]])
     
     predictors = encodedSingularsR + encodedPluralsR 
     dependent = [0 for _ in encodedSingularsR] + [1 for _ in encodedPluralsR]
     
     score = logisticRegr.score(predictors, dependent)
     print(["same length plurals", score])
     
     evaluationPoints.append(("same", score))
     
     
     
     
     # adjective plural
     
     
     adjectivePlurals = set()
     
     for sentence in training.iterator():
      for line in sentence:
        if line["posUni"] == "ADJ":
           morph = line["morph"]
           if "Number=Plur" in morph:
               adjectivePlurals.add(line["word"].lower())
               
     predictors = encodeListOfWords(["."+x for x in adjectivePlurals])
     dependent = [1 for _ in predictors]
     score = logisticRegr.score(predictors, dependent)
     print(["adjective plurals", score])
     
     # now look at other words that end in n, s, e
     
     wordsEndingIn = {"r" : set(), "s" : set(), "n" : set(), "e" : set()}
     
     for sentence in training.iterator():
      for line in sentence:
        if line["posUni"] == "NOUN":
           morph = line["morph"]
           if "Number=Plur" not in  morph and "Case=Dat" not in morph:
             if line["word"][-1] in wordsEndingIn:
               wordsEndingIn[line["word"][-1]].add(line["word"].lower())
     
     predictors = encodeListOfWords(["."+x for x in wordsEndingIn["r"]])
     dependent = [0 for _ in predictors]
     score = logisticRegr.score(predictors, dependent)
     print(["r", score])
     
     evaluationPoints.append(("r_distract", score))
     
     
     
     predictors = encodeListOfWords(["."+x for x in wordsEndingIn["s"]])
     dependent = [0 for _ in predictors]
     score = logisticRegr.score(predictors, dependent)
     print(["s", score])
     
     evaluationPoints.append(("s_distract", score))
     
     
     
     predictors = encodeListOfWords(["."+x for x in wordsEndingIn["n"]])
     dependent = [0 for _ in predictors]
     score = logisticRegr.score(predictors, dependent)
     print(["n", score])
     
     evaluationPoints.append(("n_distract", score))
     
     
     
     
     predictors = encodeListOfWords(["."+x for x in wordsEndingIn["e"]])
     dependent = [0 for _ in predictors]
     score = logisticRegr.score(predictors, dependent)
     print(["e", score])
     
     evaluationPoints.append(("e_distract", score))

print("----------------")

import math

firstEntries = list(set([x[0] for x in evaluationPoints]))
for entry in firstEntries:
   values = [x[1] for x in evaluationPoints if x[0] == entry]
   accuracy = sum(values)/len(values)
   sd = math.sqrt(sum([x**2 for x in values])/len(values) - accuracy**2)
   values = sorted(values)
   lower = values[int(0.05*len(values))]
   upper = values[int(0.95*len(values))]
   print(entry, accuracy, sd, lower, upper)


quit()


