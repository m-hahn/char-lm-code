import random
 

def load(language, partition):
  if language == "italian":
    path = "/private/home/mhahn/data/WIKIPEDIA/itwiki/itwiki-"+partition+"-tagged.txt"
  else:
    path = "/private/home/mhahn/data/WIKIPEDIA/"+language+"-"+partition+"-tagged.txt"

  chunk = []
  with open(path, "r") as inFile:
    for line in inFile:
      index = line.find("\t")
      if index == -1:
          continue
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 40000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
  yield chunk

def training(language):
  return load(language, "train")
#   with open("/private/home/mhahn/data/WIKIPEDIA/"+language+"-train.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
def dev(language):
  return load(language, "valid")
#   with open("/private/home/mhahn/data/WIKIPEDIA/"+language+"-valid.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
#

#     for line in data:
#        yield line

