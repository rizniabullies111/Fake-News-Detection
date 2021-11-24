import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


np.random.seed(500)
Corpus = pd.read_csv("final_data.csv",encoding='latin-1')
Corpus = Corpus.loc[:,['text','label']]
Corpus.dropna(inplace=True)

Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus.loc[index,'text_final'] = str(Final_words)

print(Corpus.head())

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
pickle.dump(Encoder, open('label.sav','wb'))

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
pickle.dump(Tfidf_vect, open('tfidf.sav', 'wb'))

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

pickle.dump(SVM, open('model_svm.sav', 'wb'))

predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> \n ",accuracy_score(predictions_SVM, Test_Y)*100)
print("Confusion matrix -> \n",confusion_matrix(Encoder.inverse_transform(Test_Y),Encoder.inverse_transform(predictions_SVM)))
print("Classification report -> \n",classification_report(Encoder.inverse_transform(Test_Y),Encoder.inverse_transform(predictions_SVM)))

gnb = GaussianNB()
gnb.fit(Train_X_Tfidf.toarray(),Train_Y)

pickle.dump(gnb, open('model_gnb.sav', 'wb'))

predictions_gnb = gnb.predict(Test_X_Tfidf.toarray())
print("GNB Accuracy Score -> \n ",accuracy_score(predictions_gnb, Test_Y)*100)
print("Confusion matrix -> \n",confusion_matrix(Encoder.inverse_transform(Test_Y),Encoder.inverse_transform(predictions_gnb)))
print("Classification report -> \n",classification_report(Encoder.inverse_transform(Test_Y),Encoder.inverse_transform(predictions_gnb)))