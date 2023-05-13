import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

# please use the requirements_text.txt file to install the required packages

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

data_name = "trump.txt"

TRAIN = False  # make that True if you want to train the model again
if TRAIN:
    model1 = train_word2vec(data_name)
else:
    model1 = Word2Vec.load(data_name + ".model") 

w1, w2, w3, w4 = 'china', 'mexico', 'crime', 'trade'
print(f"Cosine similarity {w1} and {w3}: {model1.wv.similarity(w1, w3)}")
print(f"Cosine similarity {w2} and {w3}: {model1.wv.similarity(w2, w3)}")
print(f"Cosine similarity {w1} and {w4}: {model1.wv.similarity(w1, w4)}")
print(f"Cosine similarity {w2} and {w4}: {model1.wv.similarity(w2, w4)}")

w1, w2, w3, w4 = 'man', 'woman', 'job', 'parent'
print(f"Cosine similarity {w1} and {w3}: {model1.wv.similarity(w1, w3)}")
print(f"Cosine similarity {w2} and {w3}: {model1.wv.similarity(w2, w3)}")
print(f"Cosine similarity {w1} and {w4}: {model1.wv.similarity(w1, w4)}")
print(f"Cosine similarity {w2} and {w4}: {model1.wv.similarity(w2, w4)}")