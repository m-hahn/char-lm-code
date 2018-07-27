import os
import random
#import accessISWOCData
#import accessTOROTData
import sys
 
import edlib

import csv

def readTSV(paths, language=None):
   result = []
   header = None
   assert len(paths) < 10
   paths = sorted(paths)
   print(paths)
   for path in paths:
      print(path)
      with open(path, "r") as inFile:
#         data = csv.reader(inFile, delimiter=",", quotechar='"')
         data = [x.split("\t") for x in inFile.read().strip().split("\n")]
 #        headerNew = data[0]
         if header is None:
            headerNew = data[0]
            data = data[1:]
            header = headerNew

         if language is not None:
            languageIndex = header.index("language")
            print(languageIndex)
            for line in data:
              assert len(line) <= len(header), (header, line)
              if len(line) > len(header):
                print((line, header))
              assert languageIndex < len(line), (header, line)
            data = [x for x in data if x[languageIndex] == language]
         result += data
         assert header == headerNew, (header, headerNew)
   return (header, result)



def readCSV(paths):
   result = []
   header = None
   assert len(paths) < 10
   paths = sorted(paths)
   print(paths)
   for path in paths:
      print(path)
      with open(path, "r") as inFile:
         data = csv.reader(inFile, delimiter=",", quotechar='"')
#         data = [x.split("\t") for x in inFile.read().strip().split("\n")]
 #        headerNew = data[0]
         if header is None:
            headerNew = next(data)
            header = headerNew

         result += list(data)
         assert header == headerNew, (header, headerNew)
   return (header, result)

def printTSV(table, path):
   header, data = table
   with open(path, "w") as outFile:
       outFile.write("\t".join(header)+"\n")
       for line in data:
           outFile.write("\t".join(line)+"\n")

class AcqdivReader():
   def __init__(self, language):
      basePath = "/private/home/mhahn/data/acqdiv-database/tsv/"+language.lower()+"/"
      self.morphemes = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("morphemes") and x.endswith(".tsv")])
#      printTSV("/private/home/mhahn/data/acqdiv-database/tsv/"

      self.speakers = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("speakers") and x.endswith(".tsv")])
      self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances") and x.endswith(".tsv")])
      self.words = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("words") and x.endswith(".tsv")])
      self.uniquespeakers = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("uniquespeakers") and x.endswith(".tsv")])
     
      random.Random(4656).shuffle(self.utterances[1])
      self.language = language
      self.UTTERANCE_COLNAME = {"Japanese" : "utterance_raw", "Indonesian" : "utterance", "Sesotho" : "utterance"}[self.language]
   
   def length(self):
      return len(self.utterances)

   def iterator(self, markUtteranceBoundaries=True, blankBeforeEOS=True):
     utterance_raw_index = self.utterances[0].index(self.UTTERANCE_COLNAME)
     for sentence in self.utterances[1]:
        yield (sentence[utterance_raw_index]+((" " if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")).lower()

   def iteratorMorph(self, markUtteranceBoundaries=True, blankBeforeEOS=True):
     utterance_raw_index = self.utterances[0].index(self.UTTERANCE_COLNAME)
     morpheme_index = self.utterances[0].index("morpheme")
     gloss_raw_index = self.utterances[0].index("gloss_raw")
     pos_raw_index = self.utterances[0].index("pos_raw")
     for sentence in self.utterances[1]:
        utterance_raw = (sentence[utterance_raw_index]).lower()
        utterance_for_return = utterance_raw+((" " if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")
        utterance = [x for x in utterance_raw.lower().split(" ") if x != ""]
        morpheme = [x for x in sentence[morpheme_index].split(" ") if x != ""]
        gloss_raw = [x for x in sentence[gloss_raw_index].split(" ") if x != ""]
        pos_raw = [x for x in sentence[pos_raw_index].split(" ") if x != ""]
        if self.language == "Japanese":
          for l in [morpheme, gloss_raw, pos_raw]:
            while len(l) < len(utterance):
              l.append("")

        # japanese: expect same length, or annotation missing altogether
        # sesotho: annotation matching, but utterance may have different tokenization --> best chance seems to be heuristic alignment
        # indonesian: some splitting of words into morphemes, with combined whitespace and "-"

        if self.language == "Sesotho":
             print(".....")
             print(utterance)
             print(morpheme)
             print(gloss_raw)
             print(pos_raw)
             if len(morpheme) != len(pos_raw):
                _ = 0
             else:
               #print(morpheme)
               #print(pos_raw)
               # align morphemes with the tags
               for j in range(len(morpheme)):
                   assert len(morpheme[j].split("-")) == len(pos_raw[j].split("-"))
               rawUtterance = "".join(utterance)
               rawMorphemes = "".join(morpheme).replace("-", "").lower()
               #print(rawUtterance)
               #print(rawMorphemes)
               cigar = edlib.align(rawMorphemes, rawUtterance, task="path")["cigar"]
               utteranceSegmentedIntoMorphemes = []
               currentUtterancePortion = ""
               positionInRawUtterance = 0
               positionInMorphemes = 0
               morphemesString = " ".join(morpheme)
              # print(cigar)
               j = 0
               while j < len(cigar):
                  k = j
                  while k < len(cigar):
                     if "0" <= cigar[k] and cigar[k] <= "9":
                        k += 1
                     else:
                            break
                  length = int(cigar[j:k])
                  action = cigar[k]
                  j = k+1
                  for k in range(length):
                  #    print((currentUtterancePortion, utteranceSegmentedIntoMorphemes, "rawUtterance", rawUtterance[positionInRawUtterance:], "morphemeString", morphemesString[positionInMorphemes:]))
                      while positionInMorphemes < len(morphemesString) and morphemesString[positionInMorphemes] in [" ","-"]:
                          utteranceSegmentedIntoMorphemes.append(currentUtterancePortion)
                          currentUtterancePortion = ""
                          positionInMorphemes+= 1

                      if action in ["X", "="]:
                          assert positionInRawUtterance < len(rawUtterance)
                          currentUtterancePortion += rawUtterance[positionInRawUtterance]
                      elif action == "D":
                          currentUtterancePortion  += rawUtterance[positionInRawUtterance]
#                      elif action == "D":
                           
                      if action != "I":
                        positionInRawUtterance += 1
                      if action != "D":
                          positionInMorphemes +=1
                            
               utteranceSegmentedIntoMorphemes.append(currentUtterancePortion)
               print(utteranceSegmentedIntoMorphemes)    # for Sesotho, now make this the tokenization
               utteranceReTokenized = []
               utteranceSegmentedIntoMorphemesFinal = []
               indexSegments = 0
               for word in morpheme:
                    components = len(word.split("-"))
                    utteranceReTokenized.append("".join(utteranceSegmentedIntoMorphemes[indexSegments:indexSegments+components]))
                    utteranceSegmentedIntoMorphemesFinal.append(utteranceSegmentedIntoMorphemes[indexSegments:indexSegments+components])
                    indexSegments += components
               utteranceSegmentedIntoMorphemes = utteranceSegmentedIntoMorphemesFinal 
               print(utteranceReTokenized)
               print(utteranceSegmentedIntoMorphemes)
               utterance = utteranceReTokenized
               assert len(utterance) == len(morpheme)
        else:
            utteranceSegmentedIntoMorphemes = [[x] for x in utterance]
#                assert pos_raw[0] == "none" and len(pos_raw) == 1
#             else:
#                assert len(morpheme) == len(pos_raw)
#        if random.random() > 0.95:
#            quit()
        annotated = list(zip(utterance, morpheme, gloss_raw, pos_raw, utteranceSegmentedIntoMorphemes))
        yield (utterance_for_return, annotated)



class AcqdivReaderPartition():
   def __init__(self, reader, partition="train"):
        self.corpus = reader
        self.partition = partition
   def reshuffledIterator(self, markUtteranceBoundaries=True, blankBeforeEOS=True, originalIterator = AcqdivReader.iterator):
      results = list(self.iterator(markUtteranceBoundaries=markUtteranceBoundaries, blankBeforeEOS=blankBeforeEOS, originalIterator = originalIterator))
      random.shuffle(results)
      for utterance in results:
        yield utterance
   def iterator(self, markUtteranceBoundaries=True, blankBeforeEOS=True, originalIterator = AcqdivReader.iterator):
        iterator = originalIterator(self.corpus, markUtteranceBoundaries=markUtteranceBoundaries, blankBeforeEOS=blankBeforeEOS)
        for _ in range(10000):
            x = next(iterator)
            if self.partition == "dev":
                 yield x
        if self.partition == "train":
          for x in iterator:
              yield x


#reader = AcqdivReaderPartition(AcqdivReader("Japanese"), "train").iterator()
#
#print(next(reader))
#

