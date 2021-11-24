import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import pickle


np.random.seed(500)
Corpus1 = pd.read_csv("test.csv",encoding='latin-1')
Corpus1['text'].dropna(inplace=True)

Corpus1['text'] = [entry.lower() for entry in Corpus1['text']]
Corpus1['text']= [word_tokenize(entry) for entry in Corpus1['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus1['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus1.loc[index,'text_final'] = str(Final_words)

Test_X1 = Corpus1['text_final']
Tfidf_vect = pickle.load(open('tfidf.sav', 'rb'))
Test_X_Tfidf1 = Tfidf_vect.transform(Test_X1)

SVM = pickle.load(open('model_svm.sav', 'rb'))
predictions_SVM = SVM.predict(Test_X_Tfidf1)

Encoder = pickle.load(open('label.sav', 'rb'))
print("output svm")
print(Encoder.inverse_transform(predictions_SVM))

gnb = pickle.load(open('model_gnb.sav', 'rb'))
predictions_gnb = SVM.predict(Test_X_Tfidf1.toarray())

print("output gnb")
print(Encoder.inverse_transform(predictions_gnb))
