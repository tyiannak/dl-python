import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize


def train_word2vec(file_name):
    sample = open(file_name)
    s = sample.read()
    # pre_process text:
    f = s.replace("\n", " ")
    data = []
    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
        data.append(temp)
    print(len(data))

    model = gensim.models.Word2Vec(data, min_count=2,
                                   vector_size=500,
                                   window=5, sg=1)
    model.save(file_name + ".model")
    print(file_name + ".model")
    return model


#data_name = "books_small.txt"
data_name = "trump.txt"

if 0:
    model1 = train_word2vec(data_name)
else:
    model1 = Word2Vec.load(data_name + ".model") 

w1 = 'china'
w2 = 'mexico'
w3 = 'crime'
w4 = 'trade'
print(f"Cosine similarity between {w1} and {w3}: {model1.wv.similarity(w1, w3)}")
print(f"Cosine similarity between {w2} and {w3}: {model1.wv.similarity(w2, w3)}")
print(f"Cosine similarity between {w1} and {w4}: {model1.wv.similarity(w1, w4)}")
print(f"Cosine similarity between {w2} and {w4}: {model1.wv.similarity(w2, w4)}")

w1 = 'man'
w2 = 'woman'
w3 = 'job'
w4 = 'parent'
print(f"Cosine similarity between {w1} and {w3}: {model1.wv.similarity(w1, w3)}")
print(f"Cosine similarity between {w2} and {w3}: {model1.wv.similarity(w2, w3)}")
print(f"Cosine similarity between {w1} and {w4}: {model1.wv.similarity(w1, w4)}")
print(f"Cosine similarity between {w2} and {w4}: {model1.wv.similarity(w2, w4)}")


#vector = model1.wv['woman']
#sims = model1.wv.most_similar('woman', topn=20)
#print(sims)