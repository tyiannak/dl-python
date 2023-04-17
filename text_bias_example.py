import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

if 0:
   sample = open("books.txt")
   s = sample.read()
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
   model1 = gensim.models.Word2Vec(data, min_count = 2, vector_size = 500, window = 5, sg = 1)
   model1.save("word2vec.model")
else:
   model1 = Word2Vec.load("word2vec.model") 

w1 = 'he'
w2 = 'she'
w3 = 'strong'
w4 = 'sweet'
print(f"Cosine similarity between {w1} and {w3}: {model1.wv.similarity(w1, w3)}")
print(f"Cosine similarity between {w2} and {w3}: {model1.wv.similarity(w2, w3)}")

print(f"Cosine similarity between {w1} and {w4}: {model1.wv.similarity(w1, w4)}")
print(f"Cosine similarity between {w2} and {w4}: {model1.wv.similarity(w2, w4)}")

vector = model1.wv['woman']
sims = model1.wv.most_similar('woman', topn=20)
print(sims)
