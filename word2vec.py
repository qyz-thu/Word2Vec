import re
import numpy as np
import copy
import time
np.seterr(divide='ignore', invalid='ignore')
layer_size = 250     # dimension of word embedding
alpha = 0.02        # learning rate
window_size = 5     # size of window
table_size = 10000000   # size of unigram table
vocab = {}          # vocabulary
cbow = True         # using cbow model. default is true
hs = True           # using hierarchical softmax. default is true
ns = 5              # number of negative samples.
theta_neg = []      # theta in negative sampling
time0 = 0           # starting time
time1 = 0           # ending time
words_total = 0     # total amount of words
MaxExp = 6          # max exp
ExpSize = 1000      # size of exp table
startPosition = 0
currentPosition = 0
expTable = []
wordsTrainedBefore = True
iterationTime = 1   # times of iteration
wordSim = []
vocFile = 'D:/nlp_reso/language model/vocabulary.txt'      # file path of vocabulary data
trainedVocFile = "D:/nlp_reso/result.txt"       # file path of trained voc
trainFile = 'D:/学习/nlp/data/word2vec/train_corpora.txt'     # file path of training data
saveFile = 'D:/nlp_reso/result.txt'             # file path of saving word embedding
wordsimFile = "D:/nlp_reso/wordsim-353.txt"     # file path of wordsim353
googleanalogyFile = "D:/nlp_reso/analogy.txt"   # file path of google analogy
sample = 1e-5       # subsampling rate


class Word:

    def __init__(self, name):
        self.s = name
        self.cn = 1
        self.points = []    # nodes in path to root
        self.code = []      # binary code in huffman tree
        self.we = np.random.uniform(-0.5 / layer_size, 0.5 / layer_size, layer_size)
        self.id = 0
        self.prob = 1       # probability of being sampled in subsampling


class Nodes:

    def __init__(self):
        self.cn = 1000000000000000
        self.theta = np.zeros((layer_size,), dtype=np.float_)
        pass


def word_similarity(w1, w2):
    if w1 not in vocab or w2 not in vocab:
        return -1
    v1 = vocab[w1].we
    v2 = vocab[w2].we
    s1 = np.vdot(v1, v2)
    s2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    ans = 5 * (s1 / s2 + 1)
    if s2 != 0:
        return ans
    else:
        return -2


def create_huffman_tree():
    vocab_size = len(vocab_list)
    for i in range(vocab_size):
        vocab_list[i].id = i
    p1 = vocab_size - 1
    p2 = vocab_size
    parents = [0 for i in range(2 * vocab_size - 1)]
    binary = [0 for i in range(2 * vocab_size - 1)]
    for j in range(vocab_size - 1):
        n = Nodes()
        vocab_list.append(n)
    for i in range(vocab_size - 1):
        # find the two smallest nodes
        if p1 < 0:
            min1 = copy.deepcopy(p2)
            p2 += 1
            min2 = copy.deepcopy(p2)
            p2 += 1
        else:
            if vocab_list[p1].cn <= vocab_list[p2].cn:
                min1 = copy.deepcopy(p1)
                p1 -= 1
            else:
                min1 = copy.deepcopy(p2)
                p2 += 1
            if p1 < 0:
                min2 = copy.deepcopy(p2)
                p2 += 1
            else:
                if vocab_list[p1].cn <= vocab_list[p2].cn:
                    min2 = copy.deepcopy(p1)
                    p1 -= 1
                else:
                    min2 = copy.deepcopy(p2)
                    p2 += 1
        vocab_list[vocab_size + i].cn = vocab_list[min1].cn + vocab_list[min2].cn
        parents[min1] = vocab_size + i
        parents[min2] = vocab_size + i
        binary[min2] = 1

    # assign parent nodes and binary code to each word
    for i in range(vocab_size):
        a = copy.deepcopy(i)
        vocab_list[i].points.append(vocab_size - 2)
        while a != (2 * vocab_size - 2):
            if a >= vocab_size:
                vocab_list[i].points.insert(1, a - vocab_size)
            vocab_list[i].code.insert(0, binary[a])
            a = parents[a]
    nodes = []
    for i in range(vocab_size, 2 * vocab_size -1):
        nodes.insert(0, vocab_list.pop())

    return nodes


def init_unigram_table():
    table = [0 for i in range(table_size)]
    _sum = 0
    for i in range(len(vocab)):
        _sum += vocab_list[i].cn**0.75
    i = 0
    cur_sum = vocab_list[i].cn**0.75
    for j in range(table_size):
        if j * _sum > table_size * cur_sum:
            i += 1
            cur_sum += vocab_list[i].cn**0.75
        table[j] = i
    return table


def save_file():
    with open(saveFile, 'w') as f_save:
        f_save.write(str(currentPosition) + '\n')
        for word in vocab:
            f_save.write(word)
            f_save.write(': [')
            for i in range(layer_size):
                f_save.write(str(vocab[word].we[i]))
                f_save.write(', ')
            f_save.write(']\n')
    print('\ndata saved.\n')


def read_wordsim():
    global wordSim
    with open(wordsimFile, 'r') as f_sim:
        for wordpairs in f_sim:
            words = re.findall('[a-zA-Z]+', wordpairs)
            words[0] = words[0].lower()
            words[1] = words[1].lower()
            num = re.search('[0-9]+', wordpairs)
            num = float(num.group(0))
            l = [words[0], words[1], num]
            wordSim.append(l)


def test_simi():
    loss = 0
    cn = 0
    for l in wordSim:
        if l[0] not in vocab or l[1] not in vocab:
            continue
        s = word_similarity(l[0], l[1])
        e = s - l[2]
        loss += e**2
        cn += 1
    loss /= cn
    loss **= 0.5
    print('\ncurrent loss: %.2f' % loss)


def train_model():
    global time0
    global time1
    global alpha
    global currentPosition
    global startPosition
    global window_size
    initial_alpha = alpha
    time0 = time.time()
    root = np.zeros(layer_size)
    e = np.zeros(layer_size)
    actual_word_count = 0
    word_count = 0
    actual_byte_count = 0
    byte_count = 0
    for iter in range(iterationTime):
        print("\n\nEpoch: %d\n\n" % iter)
        with open(trainFile, 'rb') as f_train:
            doprint = True
            for sentence in f_train:
                currentPosition += 1
                if currentPosition <= startPosition:
                    continue
                if doprint:
                    print("start line: %d" % currentPosition)
                    doprint = False
                sentence = str(sentence)
                byte_count += len(sentence)
                if byte_count >= 1000000:
                    byte_count -= 1000000
                    actual_byte_count += 1
                    if (actual_byte_count % 10) == 0:
                        print("%d M bytes counted" % actual_byte_count)
                        print("alpha = %f" % alpha)
                        time1 = time.time()
                        time_used = time1 - time0
                        h = time_used // 3600
                        time_used1 = time_used % 3600
                        min = time_used1 // 60
                        time_used2 = time_used1 % 60
                        print('time used: %d h %d min %.2f s' % (h, min, time_used2))
                    if (actual_byte_count % 100) == 0:
                        test_simi()
                        save_file()
                words = re.findall('[a-zA-Z]+', sentence)
                for pre in range(len(words) - 1, -1, -1):
                    if words[pre] not in vocab:
                        del words[pre]
                        continue
                    if vocab[words[pre]].prob < np.random.uniform():
                        del words[pre]
                for pos in range(len(words)):
                    cur_word = words[pos]
                    if word_count >= 1000000:
                        actual_word_count += 1
                        word_count = 0
                        alpha = initial_alpha * (1 - 10000 * actual_word_count / (words_total * iterationTime + 1))
                        if alpha < initial_alpha * 0.001:
                            alpha = initial_alpha * 0.001
                        print("Words trained: %d M" % actual_word_count)
                    num = 0
                    root *= 0
                    e *= 0
                    window = np.random.randint(1, window_size + 1)
                    b = np.random.randint(0, window)
                    if cbow:  # train cbow model
                        for i in range(2 * window + 1):    # construct root
                            p = pos - window + i + b
                            if p < 0 or p >= len(words):
                                continue
                            if p != pos:
                                root += vocab[words[p]].we
                                num += 1
                        if num == 0:
                            continue
                        root /= num
                        if hs:      # hierarchical softmax
                            for d in range(len(vocab[cur_word].code)):
                                index = vocab[cur_word].points[d]
                                an = nodes_list[index]
                                f = np.vdot(root, an.theta)
                                if f > MaxExp:
                                    continue
                                elif f < -MaxExp:
                                    continue
                                else:
                                    index = int(ExpSize * (f + MaxExp) / (2 * MaxExp))
                                    f = expTable[index]
                                # g is the gradient multiplied by learning rate
                                g = alpha * (1 - vocab[cur_word].code[d] - f)
                                e += g * an.theta
                                an.theta += g * root
                        if ns > 0:      # negative sampling
                            for d in range(ns + 1):
                                if d > 0:
                                    r = np.random.randint(0, table_size)
                                    target = u_table[r]
                                    if vocab_list[target].s == cur_word:
                                        continue
                                    label = 0
                                else:
                                    target = vocab[cur_word].id
                                    label = 1
                                global theta_neg
                                f = np.vdot(root, theta_neg[target])
                                if f > MaxExp:
                                    f = 1
                                elif f < -MaxExp:
                                    f = 0
                                else:
                                    index = int(ExpSize * (f + MaxExp) / (2 * MaxExp))
                                    f = expTable[index]
                                g = alpha * (label - f)
                                e += g * theta_neg[target]
                                theta_neg[target] += g * root
                        # hidden -> in
                        for i in range(2 * window + 1):
                            p = pos - window + i + b
                            if p < 0 or p >= len(words):
                                continue
                            if p != pos and words[p] in vocab:
                                vocab[words[p]].we += e
                    else:      # train skip-gram model
                        for i in range(2 * window + 1):
                            p = pos - window + i + b
                            if p < 0 or p >= len(words):
                                continue
                            c = vocab[words[p]]
                            e *= 0
                            if hs:
                                for d in range(len(vocab[cur_word].code)):
                                    f = np.vdot(c.we, nodes_list[vocab[cur_word].points[d]].theta)
                                    if f > MaxExp:
                                        continue
                                    elif f < -MaxExp:
                                        continue
                                    else:
                                        index = int(ExpSize * (f + MaxExp) / (2 * MaxExp))
                                        f = expTable[index]
                                    g = alpha * (1 - vocab[cur_word].code[d] - f)
                                    e += g * nodes_list[vocab[cur_word].points[d]].theta
                                    nodes_list[vocab[cur_word].points[d]].theta += g * c.we
                            if ns > 0:
                                for d in range(ns + 1):
                                    if d > 0:
                                        r = np.random.randint(0, table_size)
                                        target = u_table[r]
                                        if vocab_list[target].s == cur_word:
                                            continue
                                        label = 0
                                    else:
                                        target = vocab[cur_word].id
                                        label = 1
                                    f = np.vdot(c.we, theta_neg[target])
                                    if f > MaxExp:
                                        f = 1
                                    elif f < -MaxExp:
                                        f = 0
                                    else:
                                        index = int(ExpSize * (f + MaxExp) / (2 * MaxExp))
                                        f = expTable[index]
                                    g = alpha * (label - f)
                                    e += g * theta_neg[target]
                                    theta_neg[target] += g * c.we
                            c.we += e
                    word_count += 1
    time1 = time.time()
    time_used = time1 - time0
    h = time_used // 3600
    time_used1 = time_used % 3600
    min = time_used1 // 60
    time_used2 = time_used1 % 60
    print('total time used: %d h %d min %.2f s' % (h, min, time_used2))
    currentPosition = 0
    save_file()


# precompute exp table
for i in range(ExpSize):
    Exp = MaxExp * (2 * i / ExpSize - 1)
    Exp = np.exp(Exp)
    Exp /= (Exp + 1)
    expTable.append(Exp)

vocab_list = []
nodes_list = []
# read in words
read_wordsim()
with open(vocFile, 'r') as f_voc:
    words = f_voc.readlines()
    for w in words:
        word = re.search('[a-zA-Z]+', w)
        word = word.group(0)
        count = re.search('[0-9]+', w)
        count = int(count.group(0))
        if count < 100:
            break
        vocab[word] = Word(word)
        vocab[word].cn = count
        words_total += count
        vocab_list.append(vocab[word])
if wordsTrainedBefore:
    with open(trainedVocFile, 'r') as f_voc:
        words = f_voc.readlines()
        first_line = True
        for w in words:
            if first_line:
                n = re.search('[0-9]+', w)
                startPosition = int(n.group(0)) + 1
                first_line = False
                continue
            word = re.search('[a-zA-Z]+', w)
            word = word.group(0)
            vocab[word].prob = (vocab[word].cn / (sample * words_total) + 1) ** 0.5
            vocab[word].prob *= sample * words_total / vocab[word].cn
            if word not in vocab:
                continue
            nums = re.findall('[0-9-]+[0-9.e-]+', w)
            i = 0
            for num in nums:
                vocab[word].we[i] = float(num)
                i += 1

nodes_list = create_huffman_tree()
if ns > 0:
    u_table = init_unigram_table()
    for i in range(len(vocab)):
        vec = np.zeros(layer_size)
        theta_neg.append(vec)
train_model()


