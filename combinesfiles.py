from os import listdir
from os.path import isfile, join
mypath = "train"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
outercount = 1
ignorecount = 1
total_sentence_count = 1

with open('temp.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("Sentence #,Word,POS,Tag\n")
    for fname in onlyfiles:
        with open(mypath +"/" + fname, encoding="utf-8")as infile:
            for line in infile:
                wordconnect = ''
                if line == '\n':
                    ignorecount = 1
                    continue
                if line.split()[0] == '-DOCSTART-':
                    continue
                if ignorecount == 1:
                    wordconnect = "Sentence: " + str(total_sentence_count)
                    total_sentence_count += 1
                    ignorecount = -1
                splitWords = line.split('\t')
                if splitWords[0] == ',':
                    continue
                else:
                    wordconnect += '\t' + str(splitWords[0]) + '\t' + splitWords[1] + '\t' + splitWords[3].replace('\n','') + '\n'
                outfile.write(wordconnect)


