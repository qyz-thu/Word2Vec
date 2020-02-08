import re
dictionary = 'D:/nlp_reso/language model/Dictionary.txt'
readFile = 'D:/学习/nlp/data/word2vec/train_corpora.txt'
saveFile = 'D:/nlp_reso/language model/vocabulary.txt'
vocab = {}
actual_word = {}
FrequencyThreshold = 100    # words with less frequency will be discarded


class Words:

    def __init__(self, word):
        self.s = word
        self.cn = 1


with open(dictionary, 'r') as f:
    content = f.read()
wordset = re.findall('[a-z]+', content)
for w in wordset:
    actual_word[w] = 0

# read words from data
with open(readFile, 'rb') as f:
    i = 0
    j = 0
    for line in f:
        line = str(line)
        i += 1
        if i >= 1000:
            j += 1
            i = 0
            print('read %d k lines' % j)
        words = re.findall('[a-zA-Z]+', line)
        for word in words:
            if word in vocab:
                vocab[word].cn += 1
            else:
                if word in actual_word:
                    w = Words(word)
                    vocab[word] = w

# sort
voc = []
for word in vocab:
    voc.append(vocab[word])
voc.sort(key=lambda a: a.cn, reverse=True)
# save words
with open(saveFile, 'w') as f:
    for word in voc:
        if word.cn < FrequencyThreshold:
            break
        f.write(word.s)
        f.write(' ')
        f.write(str(word.cn))
        f.write('\n')
