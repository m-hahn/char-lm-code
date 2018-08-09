
sentences = [0,0]

with open("/private/home/mhahn/data/WIKIPEDIA/german-valid-tagged.txt", "r") as inFile:
   with open("/checkpoint/mhahn/german-valid-500.txt", "w") as outFile:
      currentLine = []
      currentLineCharCount = 0
      for line in inFile:
          i = line.find("\t")
          if i == -1:
             continue
          word = line[:i].lower()
 #         print(f"#{word}#")
#          print([line[i+1:i+3]])
          currentLine.append(word)
          currentLineCharCount += len(word) + 1
          if word == "." and line[i+1:i+3] == "$.":
              if currentLineCharCount > 450:
                print("ERROR "+str(currentLineCharCount))
                print((currentLine))
                sentences[1] += 1
              else:
                print(" ".join(currentLine), file=outFile)
                sentences[0] += 1
              currentLine = []
              currentLineCharCount = 0
print(sentences)
