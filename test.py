import numpy as np
import re
from scipy.stats import spearmanr
import time
layer_size = 250
vocab = {}
filename = 'D:/nlp_reso/result.txt'
wordsimFile = "D:/nlp_reso/wordsim-353.txt"
googleanalogyFile = "D:/nlp_reso/analogy.txt"
saveFile = "D:/nlp_reso/test_result.txt"


class Word:

    def __init__(self, word):
        self.s = word
        self.we = np.zeros(layer_size)


def word_similarity(w1, w2):
    if w1 not in vocab or w2 not in vocab:
        return -1
    v1 = vocab[w1].we
    v2 = vocab[w2].we
    s1 = np.vdot(v1, v2)
    s2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    ans = 5 * (s1 / s2 + 1)
    print("the similarity between %s and %s is " % (w1, w2), end='')
    if s2 != 0:
        print(ans, end='\t')
        return ans
    else:
        return -2


def find_similar_words(w, num=3):
    if w not in vocab:
        print("Word not in vocabulary.")
        return
    print("%d most similar words to %s are:" % (num, w))
    v1 = vocab[w].we
    similarity = [-1 for i in range(num)]
    similar_words = ['' for i in range(num)]
    for word in vocab:
        if word != w:
            v2 = vocab[word].we
            s1 = np.vdot(v1, v2)
            s2 = np.linalg.norm(v1) * np.linalg.norm(v2)
            simi = s1 / s2
            for j in range(num):
                if simi > similarity[j]:
                    similarity[j] = simi
                    similar_words[j] = word
                    break
    for i in range(num):
        pass
        print(similar_words[i], ' ', similarity[i])


def find_word(we, w1, w2, w3, num=3):
    n1 = np.linalg.norm(we)
    similarity = [-1 for i in range(num)]
    similar_words = ['' for i in range(num)]
    for w in vocab:
        if w == w1 or w == w2 or w == w3:
            continue
        v = vocab[w].we
        f = np.vdot(v, we)
        n2 = np.linalg.norm(v)
        s = n1 * n2
        a = f / s
        for j in range(num):
            if a > similarity[j]:
                similarity[j] = a
                similar_words[j] = w
                break
    return similar_words


def test():
    word_similarity('us', 'we')
    word_similarity('good', 'better')
    word_similarity('stock', 'stocks')
    word_similarity('say', 'speak')
    word_similarity('china', 'beijing')
    word_similarity('city', 'citizen')
    word_similarity('develop', 'development')
    word_similarity('russia', 'china')
    word_similarity('scotland', 'england')
    word_similarity('japan', 'tokyo')
    word_similarity('sing', 'dance')
    word_similarity('merry', 'christmas')
    word_similarity('car', 'plane')
    word_similarity('bus', 'subway')
    word_similarity('car', 'truck')
    word_similarity('table', 'desk')
    word_similarity('merry', 'happy')
    word_similarity('love', 'sex')
    word_similarity('president', 'potato')
    word_similarity('submarine', 'ginger')

    find_similar_words('economy')
    find_similar_words('china')
    find_similar_words('trade')
    find_similar_words('machine')
    find_similar_words('stock')
    find_similar_words('cash')
    find_similar_words('turing')
    find_similar_words('twitter')
    find_similar_words('youtube')
    find_similar_words('computer')
    find_similar_words('football')


with open(filename, 'r') as f:
    wordList = f.readlines()
    first = True
    for words in wordList:
        if first:
            first = False
            continue
        word = re.search('\w+', words)
        w = Word(word.group(0))
        vocab[word.group(0)] = w
        numbers = re.findall('[0-9|\.|-]+', words)
        i = 0
        if w.s == 'th':
            pass
        for i in range(layer_size):
            w.we[i] = float(numbers[i])

wordsim = True
analogy = True
with open(saveFile, 'w') as f_w:
    time0 = time.time()
    v1 = []
    v2 = []
    avg1 = 0
    avg2 = 0
    if wordsim:
        f_w.write('wordsim test:\n')
        with open(wordsimFile, 'r') as f_s:
            total_word = 0
            correct_word = 0
            for pairs in f_s:
                words = re.findall('[a-zA-Z]+', pairs)
                value = re.search('[0-9.]+', pairs)
                value = float(value.group(0))
                res = word_similarity(words[0].lower(), words[1].lower())
                if res < 0:
                    print('word not found')
                    continue
                v1.append(value)
                v2.append(res)
                f_w.write(words[0].lower() + ' ' + words[1].lower() + ' ' + str(res) + '\n')
                print('error = %.2f' % (value - res))
                if abs(value - res) < 1:
                    correct_word += 1
                total_word += 1
            print("accuracy: %d/%d" % (correct_word, total_word))

        correlation = spearmanr(v1, v2)
        c = correlation[0]
        f_w.write("correlation coefficient is %.3f\n" % c)
        print("correlation coefficient is %.3f" % c)

        time1 = time.time()
        print(time1 - time0)

    if analogy:
        time0 = time.time()
        f_w.write('analogy test:\n')
        with open(googleanalogyFile, 'r') as f_a:
            total_word = 0
            correct_word = 0
            close_word = 0
            count = 0
            for line in f_a:
                words = re.findall('[a-zA-Z]+', line)
                if len(words) != 4:
                    continue
                for i in range(len(words)):
                    words[i] = words[i].lower()
                if words[0] not in vocab or words[1] not in vocab or words[2] not in vocab:
                    continue
                total_word += 1
                we = vocab[words[1]].we - vocab[words[0]].we + vocab[words[2]].we
                ans = find_word(we, words[0], words[1], words[2])
                if total_word % 100 == 0:
                    print(words[0] + ' ' + words[1] + ' ' + words[2] + ' ', end='')
                if total_word % 100 == 0:
                    for a in ans:
                        print(a + '\t', end='')
                notfound = True
                for i in range(len(ans)):
                    if ans[i] == words[3]:
                        close_word += 1
                        if i == 0:
                            correct_word += 1
                            count += 1
                        if total_word % 100 == 0:
                            print('\tcorrect', end=' ')
                            print(count)
                            count = 0
                        notfound = False
                        break
                if total_word % 100 == 0 and notfound:
                    print('\tincorrect', end=' ')
                    print(count)
                    count = 0

            print('accuracy: %d/%d' % (correct_word, total_word))
            f_w.write('accuracy: %d/%d\n' % (correct_word, total_word))
            print('close: %d/%d' % (close_word, total_word))
            f_w.write('close: %d/%d\n' % (close_word, total_word))
            time1 = time.time()
            print(time1 - time0)
