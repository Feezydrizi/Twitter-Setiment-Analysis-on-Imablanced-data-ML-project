import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from scikitplot.metrics import plot_confusion_matrix, roc_curve, auc
from wordcloud import WordCloud,STOPWORDS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, SpatialDropout1D

import warnings

warnings.filterwarnings("ignore")
imdb = pd.read_csv('IMDB Dataset.csv')
imdb.head()
imdb.sentiment.value_counts()
data = pd.read_csv('Tweets.csv')
# Drop neutral tweets
data = data[data['airline_sentiment'] != 'neutral']
data.head()
# Check if any null values
data.isnull().sum() * 100 / len(data)
#Sentiment classes count
airline_class=data['airline_sentiment'].value_counts()
print('Count of Sentiment classes :\n',airline_class)
#Percentage of Sentiment classes count
per_sentiment_class=data['airline_sentiment'].value_counts()/len(data)*100
print('percentage of Sentiment classes :\n',per_sentiment_class)

#Countplot and violin plot for Sentiment classes
fig,ax=plt.subplots(1,2,figsize=(16,5))
sns.countplot(data.airline_sentiment.values,ax=ax[0],palette='Set2')
#sns.violinplot(x=data.airline_sentiment.values,y=data.index.values,ax=ax[1],palette='Set2')
sns.stripplot(x=data.airline_sentiment.values,y=data.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='husl')
ax[0].set_xlabel('Review_Sentiment')
#ax[1].set_xlabel('Review_Sentiment')
#ax[1].set_ylabel('Index')
#Sentiment classes count
imdb_class=imdb['sentiment'].value_counts()
print('Count of Sentiment classes :\n',imdb_class)
#Percentage of Sentiment classes count
per_sentiment_class=imdb['sentiment'].value_counts()/len(imdb)*100
print('percentage of Sentiment classes :\n',per_sentiment_class)

#Countplot and violin plot for Sentiment classes
fig,ax=plt.subplots(1,2,figsize=(16,5))
sns.countplot(imdb.sentiment.values,ax=ax[0],palette='Set2')
sns.violinplot(x=imdb.sentiment.values,y=imdb.index.values,ax=ax[1],palette='Set2')
sns.stripplot(x=imdb.sentiment.values,y=imdb.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='husl')
ax[0].set_xlabel('Review_Sentiment')
#ax[1].set_xlabel('Review_Sentiment')
#ax[1].set_ylabel('Index')
print(data.airline.value_counts())
sns.set()
plt.figure(figsize=(7,7))
sns.countplot(y=data["airline"],palette="Set2")
plt.title("Airlines Dist.")
plt.show()
j=1
plt.subplots(figsize=(10,10),tight_layout=True)
for i in data["airline"].unique():
        x = data[data["airline"]==i]
        plt.subplot(2, 3, j)
        sns.countplot(x["airline_sentiment"],palette="Set2")
        plt.xticks(rotation=45)
        plt.title(i)
        j +=1
plt.show()
# Download wordnet & punkt from nltk library
nltk.download('punkt')
nltk.download('wordnet')
lemma = nltk.WordNetLemmatizer()
def preprocess(x):
    x = str(x)
    x = re.sub("[^a-zA-z]", " ",x)
    x = x.lower()
    x = nltk.word_tokenize(x)
    x = [lemma.lemmatize(i) for  i in x]
    x = " ".join(x)
    return x

data.text = data.text.apply(preprocess)
imdb.review = imdb.review.apply(preprocess)
# Encode categorical columns
data['sentiment']=data['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
imdb['sentiment']=imdb['sentiment'].apply(lambda x: 0 if x=='negative' else 1)
allcomments = " ".join(data.text)
wordcloud = WordCloud(width = 1200, height = 1200,
                    background_color ='white',
                    stopwords = STOPWORDS,
                    min_font_size = 12).generate(allcomments)

# plot the WordCloud image
plt.figure(figsize = (8, 8))
plt.imshow(wordcloud)
plt.title("All Tweets Wordcount")
plt.axis('off')
plt.show()
print(data.negativereason.value_counts())
plt.figure(figsize=(25,5))
sns.countplot(data.negativereason, palette="Set2")
max_features= 1500
# Apply vectorization on Airline data
vectorizer1 = sklearn.feature_extraction.text.TfidfVectorizer(min_df=10, max_features=max_features)
X = vectorizer1.fit_transform(data.text)

# Apply Normalization using TfidfTransformer
X = TfidfTransformer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,data.sentiment,test_size=0.2, random_state=0)
smote = SMOTE(random_state=777,k_neighbors=5)
# Balance the train dataset
X_train_smote,y_train_smote = smote.fit_resample(X_train, y_train)
# Apply vectorization on IMDB data
vectorizer2 = sklearn.feature_extraction.text.TfidfVectorizer(min_df=10, max_features=max_features)
IMDB_X = vectorizer2.fit_transform(imdb.review)

# Apply Normalization using TfidfTransformer
IMDB_X = TfidfTransformer().fit_transform(IMDB_X)

IMDB_X_train,IMDB_X_test,IMDB_y_train,IMDB_y_test = train_test_split(IMDB_X,imdb.sentiment,test_size=0.2, random_state=0)
SVM Model
# Initialize model
svm = LinearSVC()
svm_y_axis = list()
Training SVM and evaluating accurary of trained model on unbalanced Airline data
# Fit SVM model
svm.fit(X_train, y_train)
LinearSVC()
Model performance on test data
score = svm.score(X_test, y_test)
svm_y_axis.append(score)
print(round(score,3))
0.924
#Plot the confusion matrix
plot_confusion_matrix(y_test, svm.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, svm.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.94      0.97      0.95      1839
           1       0.86      0.75      0.80       470

    accuracy                           0.92      2309
   macro avg       0.90      0.86      0.88      2309
weighted avg       0.92      0.92      0.92      2309

Training SVM on IMDB dataset & testing on Airline dataset
svm = LinearSVC()
# Training on IMDB data
svm.fit(IMDB_X_train,IMDB_y_train)
LinearSVC()
# Testing on Airline data
score = svm.score(X_test, y_test)
svm_y_axis.append(score)
print(round(score,3))
0.559
#Plot the confusion matrix
plot_confusion_matrix(y_test, svm.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, svm.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.82      0.58      0.68      1839
           1       0.23      0.49      0.31       470

    accuracy                           0.56      2309
   macro avg       0.52      0.53      0.49      2309
weighted avg       0.70      0.56      0.60      2309

Insights:

Precision & Recall are very low for class 1 than for class 0.
It seems the model trained on IMDB data is facing difficulty in predicting class 1 correctly when it was tested on Airline data.
Training SVM and evaluating accurary of trained model on balanced Airline Dataset
svm = LinearSVC()
# Fit on balanced Airline data
svm.fit(X_train_smote, y_train_smote)
LinearSVC()
Model performance on test data
score = svm.score(X_test, y_test)
svm_y_axis.append(score)
print(round(score,3))
0.915
#Plot the confusion matrix
plot_confusion_matrix(y_test, svm.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, svm.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.95      0.94      0.95      1839
           1       0.78      0.81      0.80       470

    accuracy                           0.92      2309
   macro avg       0.87      0.88      0.87      2309
weighted avg       0.92      0.92      0.92      2309

sns.set()
plt.figure(figsize=(7,7))
sns.pointplot(x=["Default Model", "Using IMDB Dataset", "Class Balancing"], y=svm_y_axis,palette="Set2")
plt.title("SVM Prediction Analysis")
plt.show()

MultiNomialNB model
mnb_y_axis = list()
Training MultiNomial and evaluating accurary of trained model on unbalanced airline dataset
gnb = MultinomialNB()
gnb.fit(X_train.todense(), y_train)
MultinomialNB()
# sparse matrix -> dense/numpy matrix
Model performance on Test data.
score = gnb.score(X_test.todense(), y_test)
mnb_y_axis.append(score)
print(round(score,3))
0.9
#Plot the confusion matrix
plot_confusion_matrix(y_test, gnb.predict(X_test.todense()), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, gnb.predict(X_test.todense()))
print(class_report)
              precision    recall  f1-score   support

           0       0.89      0.99      0.94      1839
           1       0.96      0.53      0.68       470

    accuracy                           0.90      2309
   macro avg       0.93      0.76      0.81      2309
weighted avg       0.91      0.90      0.89      2309

Finetuning Multinomial Naive Bayes classifier pretrained on IMDB dataset & testing on Airline dataset
mnb = MultinomialNB()
mnb.fit(IMDB_X_train.todense(), IMDB_y_train)
MultinomialNB()
Model performance on Airline data
score = mnb.score(X_test.todense(), y_test)
mnb_y_axis.append(score)
print(round(score,3))
0.479
#Plot the confusion matrix
plot_confusion_matrix(y_test, mnb.predict(X_test.todense()), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, mnb.predict(X_test.todense()))
print(class_report)
              precision    recall  f1-score   support

           0       0.82      0.44      0.58      1839
           1       0.22      0.62      0.33       470

    accuracy                           0.48      2309
   macro avg       0.52      0.53      0.45      2309
weighted avg       0.70      0.48      0.52      2309

Insights:

MultinomialNB model performance is not better than SVM.
Precision & Recall are very low for class 1 than for class 0.
It seems the model trained on IMDB data is facing difficulty in predicting class 1 correctly when it was tested on Airline data.
Training MultinomialNB on balanced Airline Dataset and evaluating accurary of trained model
gnb = MultinomialNB()
gnb.fit(X_train_smote.todense(), y_train_smote)
MultinomialNB()
Model performance on test dat
score = gnb.score(X_test.todense(), y_test)
mnb_y_axis.append(score)
print(round(score,3))
0.903
#Plot the confusion matrix
plot_confusion_matrix(y_test, gnb.predict(X_test.todense()), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, gnb.predict(X_test.todense()))
print(class_report)
              precision    recall  f1-score   support

           0       0.95      0.92      0.94      1839
           1       0.73      0.82      0.77       470

    accuracy                           0.90      2309
   macro avg       0.84      0.87      0.86      2309
weighted avg       0.91      0.90      0.90      2309

sns.set()
plt.figure(figsize=(7,7))
sns.pointplot(x=["Default Model", "Using IMDB Dataset", "Class Balancing"], y=mnb_y_axis,palette="Set2")
plt.title("MultinomialNB Prediction Analysis")
plt.show()

RandomForestClassifier
rbc_y_axis = list()
Training RandomForestClassifier on unbalanced Airline dataset and evaluating accurary of trained model
rbc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rbc.fit(X_train, y_train)
RandomForestClassifier(max_depth=10, random_state=0)
Model performance on test data
score = rbc.score(X_test, y_test)
rbc_y_axis.append(score)
print(round(score,3))
0.804
#Plot the confusion matrix
plot_confusion_matrix(y_test, rbc.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, rbc.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.80      1.00      0.89      1839
           1       1.00      0.04      0.07       470

    accuracy                           0.80      2309
   macro avg       0.90      0.52      0.48      2309
weighted avg       0.84      0.80      0.72      2309

Insights:

Recall is ~ 0 for class 1, means the Random Forest model not at all learning class 1 samples.
It seems Random Foreset is very sensitive to the unbalanced data.
Training rbc on IMDB dataset & testing on Airline dataset
rbc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
# Training on IMDB dataset 
rbc.fit(IMDB_X_train, IMDB_y_train)
RandomForestClassifier(max_depth=10, random_state=0)
Model performance on Airline dataset
score = rbc.score(X_test, y_test)
rbc_y_axis.append(score)
print(round(score, 3))
0.217
#Plot the confusion matrix
plot_confusion_matrix(y_test, rbc.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, rbc.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.97      0.02      0.04      1839
           1       0.21      1.00      0.34       470

    accuracy                           0.22      2309
   macro avg       0.59      0.51      0.19      2309
weighted avg       0.81      0.22      0.10      2309

Insights:

It seems Random Forest is the worst performer among all models trained on IMDB data & tested on Airline data.
Training RandomForestClassifier and evaluating accurary of trained model on balanced airline dataset
rbc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rbc.fit(X_train_smote, y_train_smote)
RandomForestClassifier(max_depth=10, random_state=0)
Model performance on test data
score = rbc.score(X_test, y_test)
rbc_y_axis.append(score)
print(round(score,3))
0.878
#Plot the confusion matrix
plot_confusion_matrix(y_test, rbc.predict(X_test), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test, rbc.predict(X_test))
print(class_report)
              precision    recall  f1-score   support

           0       0.93      0.92      0.92      1839
           1       0.69      0.73      0.71       470

    accuracy                           0.88      2309
   macro avg       0.81      0.82      0.82      2309
weighted avg       0.88      0.88      0.88      2309

sns.set()
plt.figure(figsize=(7,7))
sns.pointplot(x=["Default Model", "Using IMDB Dataset", "Class Balancing"], y=rbc_y_axis,palette="Set2")
plt.title("RandomForestClassifier Prediction Analysis")
plt.show()

LSTM (Long Short Term Memory) model
data = pd.read_csv('Tweets.csv')
data["sentiment"]=data['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
data = data[["text", "sentiment"]]
data.head()
text	sentiment
0	@VirginAmerica What @dhepburn said.	1
1	@VirginAmerica plus you've added commercials t...	1
2	@VirginAmerica I didn't today... Must mean I n...	1
3	@VirginAmerica it's really aggressive to blast...	0
4	@VirginAmerica and it's a really big bad thing...	0
def clean_data(x):
    text = x
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text
data["text"] = data.text.apply(lambda x : clean_data(x))
data.head()
text	sentiment
0	virginamerica what dhepburn said	1
1	virginamerica plus youve added commercials to ...	1
2	virginamerica i didnt today must mean i need t...	1
3	virginamerica its really aggressive to blast o...	0
4	virginamerica and its a really big bad thing a...	0
imdb["review"] = imdb.review.apply(lambda x : clean_data(x))
imdb.head()
review	sentiment
0	one of the other reviewer ha mentioned that af...	1
1	a wonderful little production br br the filmin...	1
2	i thought this wa a wonderful way to spend tim...	1
3	basically there s a family where a little boy ...	0
4	petter mattei s love in the time of money is a...	1
Tokenize the text
X = token.texts_to_sequences(data["text"].values)
X = pad_sequences(X, maxlen=max_len)
Y = pd.get_dummies(data["sentiment"]).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Apply SMOTE to balance the dataset
X_train_smote,y_train_smote = smote.fit_resample(X_train, y_train)

# IMDB data
token.fit_on_texts(imdb['review'].values)
IMDB_X_train = token.texts_to_sequences(imdb['review'].values)
IMDB_X_train = pad_sequences(IMDB_X_train, maxlen=max_len)
IMDB_y_train = pd.get_dummies(imdb["sentiment"]).values
max_len = 100
max_features = 1500
token = Tokenizer(num_words=max_features, split = ' ')
​
# Airline data
token.fit_on_texts(data['text'].values)
X = token.texts_to_sequences(data["text"].values)
X = pad_sequences(X, maxlen=max_len)
Y = pd.get_dummies(data["sentiment"]).values
​
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Apply SMOTE to balance the dataset
X_train_smote,y_train_smote = smote.fit_resample(X_train, y_train)
​
# IMDB data
token.fit_on_texts(imdb['review'].values)
IMDB_X_train = token.texts_to_sequences(imdb['review'].values)
IMDB_X_train = pad_sequences(IMDB_X_train, maxlen=max_len)
IMDB_y_train = pd.get_dummies(imdb["sentiment"]).values
Build LSTM model
embed_dim = 500
lstm_out = 100
​
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
​
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 500)          750000    
_________________________________________________________________
spatial_dropout1d (SpatialDr (None, 100, 500)          0         
_________________________________________________________________
lstm (LSTM)                  (None, 100)               240400    
_________________________________________________________________
dense (Dense)                (None, 2)                 202       
=================================================================
Total params: 990,602
Trainable params: 990,602
Non-trainable params: 0
_________________________________________________________________
lstm_accuracy = list()
LSTM model trained on airline dataset
batch_size = 16
history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=2)
Epoch 1/10
732/732 - 146s - loss: 0.4184 - accuracy: 0.8092
Epoch 2/10
732/732 - 145s - loss: 0.3255 - accuracy: 0.8592
Epoch 3/10
732/732 - 146s - loss: 0.2841 - accuracy: 0.8804
Epoch 4/10
732/732 - 150s - loss: 0.2556 - accuracy: 0.8918
Epoch 5/10
732/732 - 152s - loss: 0.2277 - accuracy: 0.9065
Epoch 6/10
732/732 - 159s - loss: 0.2004 - accuracy: 0.9164
Epoch 7/10
732/732 - 150s - loss: 0.1725 - accuracy: 0.9283
Epoch 8/10
732/732 - 147s - loss: 0.1475 - accuracy: 0.9404
Epoch 9/10
732/732 - 149s - loss: 0.1300 - accuracy: 0.9459
Epoch 10/10
732/732 - 148s - loss: 0.1058 - accuracy: 0.9580
Model performance on test data
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
lstm_accuracy.append(acc)
print('Accuracy', round(acc,3))
183/183 - 6s - loss: 0.6863 - accuracy: 0.8200
Accuracy 0.82
y_pred = model.predict(X_test)
y_prd = y_pred.argmax(axis=1)
y_prd
array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
y_test_lstm = y_test.argmax(axis=1)
y_test_lstm
array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
#Plot the confusion matrix
plot_confusion_matrix(y_test_lstm, y_prd , normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test_lstm, y_prd)
print(class_report)
              precision    recall  f1-score   support

           0       0.86      0.85      0.86      1870
           1       0.74      0.76      0.75      1058

    accuracy                           0.82      2928
   macro avg       0.80      0.81      0.81      2928
weighted avg       0.82      0.82      0.82      2928

LSTM model pretrained on IMDB dataset and finetuned on airline dataset
batch_size = 16
history = model.fit(IMDB_X_train, IMDB_y_train, epochs=10, batch_size=batch_size, verbose=2)
Epoch 1/10
3125/3125 - 645s - loss: 0.4734 - accuracy: 0.7638
Epoch 2/10
3125/3125 - 1002s - loss: 0.3325 - accuracy: 0.8540
Epoch 3/10
3125/3125 - 1452s - loss: 0.2997 - accuracy: 0.8708
Epoch 4/10
3125/3125 - 968s - loss: 0.2748 - accuracy: 0.8843
Epoch 5/10
3125/3125 - 5659s - loss: 0.2527 - accuracy: 0.8927
Epoch 6/10
3125/3125 - 619s - loss: 0.2341 - accuracy: 0.9032
Epoch 7/10
3125/3125 - 4362s - loss: 0.2164 - accuracy: 0.9098
Epoch 8/10
3125/3125 - 953s - loss: 0.2005 - accuracy: 0.9174
Epoch 9/10
3125/3125 - 2360s - loss: 0.1942 - accuracy: 0.9210
Epoch 10/10
3125/3125 - 666s - loss: 0.1813 - accuracy: 0.9257
Model performance on airline data
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
lstm_accuracy.append(acc)
print('Accuracy', round(acc,3))
183/183 - 6s - loss: 1.1807 - accuracy: 0.5359
Accuracy 0.536
y_pred = model.predict(X_test)
#Plot the confusion matrix
plot_confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(class_report)
              precision    recall  f1-score   support

           0       0.66      0.56      0.61      1870
           1       0.39      0.50      0.44      1058

    accuracy                           0.54      2928
   macro avg       0.53      0.53      0.52      2928
weighted avg       0.56      0.54      0.54      2928

LSTM model trained on balanced airline dataset
batch_size = 16
history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=2)
Epoch 1/10
732/732 - 162s - loss: 0.4176 - accuracy: 0.8168
Epoch 2/10
732/732 - 161s - loss: 0.2704 - accuracy: 0.8853
Epoch 3/10
732/732 - 161s - loss: 0.2280 - accuracy: 0.9048
Epoch 4/10
732/732 - 169s - loss: 0.1914 - accuracy: 0.9237
Epoch 5/10
732/732 - 169s - loss: 0.1682 - accuracy: 0.9290
Epoch 6/10
732/732 - 180s - loss: 0.1500 - accuracy: 0.9390
Epoch 7/10
732/732 - 289s - loss: 0.1288 - accuracy: 0.9498
Epoch 8/10
732/732 - 294s - loss: 0.1150 - accuracy: 0.9546
Epoch 9/10
732/732 - 294s - loss: 0.0971 - accuracy: 0.9629
Epoch 10/10
732/732 - 294s - loss: 0.0889 - accuracy: 0.9671
Model performance on test data
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
lstm_accuracy.append(acc)
print('Accuracy', round(acc,3))
183/183 - 12s - loss: 0.7257 - accuracy: 0.8156
Accuracy 0.816
y_pred = model.predict(X_test)
#Plot the confusion matrix
plot_confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize=False,figsize=(12,6))
<AxesSubplot:title={'center':'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>

# Classification report
class_report= classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(class_report)
              precision    recall  f1-score   support

           0       0.86      0.85      0.85      1870
           1       0.74      0.76      0.75      1058

    accuracy                           0.82      2928
   macro avg       0.80      0.80      0.80      2928
weighted avg       0.82      0.82      0.82      2928

sns.set()
plt.figure(figsize=(7,7))
sns.pointplot(x=["Default Model", "Using IMDB Dataset", "Class Balancing"], y=lstm_accuracy,palette="Set2")
plt.title("LSTM Prediction Analysis")
plt.show()

​
