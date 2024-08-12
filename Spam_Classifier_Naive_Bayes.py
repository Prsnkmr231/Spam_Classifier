#importing all the required libraries


import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Reading the data from the csv file

messages = pd.read_csv("/content/spam.csv", encoding='ISO-8859-1')
messages = messages.drop(columns= ["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
messages.columns = ["label","message"]


#Data Cleaning and preprocessing 

corpus = []
ps = PorterStemmer()
wl = WordNetLemmatizer()
for index in range(len(messages)):
  sent = re.sub("[^a-zA-Z]"," ",messages["message"][index])
  sent = sent.lower()
  words = [wl.lemmatize(word) for word in nltk.word_tokenize(sent) if word not in set(stopwords.words('english'))]
  # words = [ps.stem(word) for word in nltk.word_tokenize(sent) if word not in set(stopwords.words('english'))]

  stem_sent = " ".join(words)
  corpus.append(stem_sent)
# messages.head()

# cv = CountVectorizer(max_features=2500)
# X = cv.fit_transform(corpus).toarray()

termv = TfidfVectorizer(max_features=2500)
X = termv.fit_transform(corpus).toarray()

Y= pd.get_dummies(messages["label"])
Y=Y.iloc[:,1].values

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)

Spam_classifier = MultinomialNB()
Spam_classifier.fit(X_train,y_train)

y_pred = Spam_classifier.predict(X_test)


confusion = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print(confusion)
print(accuracy)
