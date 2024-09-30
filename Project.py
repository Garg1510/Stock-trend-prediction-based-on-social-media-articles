# %%
"""
# Data Exploration
"""

# %%
import numpy as np
import pandas as pd
import re
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %%
tweet_collection = pd.read_csv('data.csv', names=['ID', 'Label', 'Tweet', 'Date'], header=1)
print(tweet_collection.shape)
tweet_collection.head()

# %%
tweet_collection['Label'].value_counts(normalize = True)

# %%
fig1, ax1 = plt.subplots(figsize=(5,5))

labels = ['positive', 'negative']
sizes = [len(tweet_collection[tweet_collection['Label'] == 'positive']), len(tweet_collection[tweet_collection['Label'] == 'negative'])]
explode = (0, 0.1)

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax1.axis('equal')
ax1.set_title("Original Data Set", fontsize=14)
    
plt.show()

# %%
positive_comments = tweet_collection[tweet_collection['Label'] == "positive"]
negative_comments = tweet_collection[tweet_collection['Label'] == "negative"]

# %%
#!pip install WordCloud
from wordcloud import WordCloud
from wordcloud import STOPWORDS


# %%
sentences = tweet_collection['Tweet'].tolist()
joined_sentences = ' '.join(sentences)

# %%
pos_sentences = positive_comments['Tweet'].tolist()
stop_words = ['Tesla', 'TSLA', 'twitter', 'https', 'elonmusk', 'TSLAQ', 'will', 'pic'] + list(STOPWORDS)
joined_pos_sentences = ' '.join(pos_sentences)
plt.figure(figsize=(15,15))
plt.imshow(WordCloud(stopwords = stop_words).generate(joined_pos_sentences))

# %%
neg_sentences = negative_comments['Tweet'].tolist()
stop_words = ['Tesla', 'TSLA', 'twitter', 'https', 'elonmusk', 'TSLAQ', 'will', 'pic'] + list(STOPWORDS)
joined_neg_sentences = ' '.join(neg_sentences)
plt.figure(figsize=(15,15))
plt.imshow(WordCloud(stopwords = stop_words).generate(joined_neg_sentences))

# %%
"""
# Training and Test Set
"""

# %%
tweet_df = tweet_collection[:5000]
randomized_collection = tweet_df.sample(frac=1, random_state=3)

training_test_index = round(len(randomized_collection) * 0.8)

training_set = randomized_collection[:training_test_index].reset_index(drop=True)
test_set = randomized_collection[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)

# %%
print(training_set['Label'].value_counts(normalize = True))
test_set['Label'].value_counts(normalize = True)

# %%
fig2, ax2 = plt.subplots(figsize=(5,5))

labels = ['positive', 'negative']
sizes = [len(training_set[training_set['Label'] == 'positive']), len(training_set[training_set['Label'] == 'negative'])]
explode = (0, 0.1)

ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax2.axis('equal')
ax2.set_title("Training Set", fontsize=14)
    
plt.show()

# %%
fig3, ax3 = plt.subplots(figsize=(5,5))

labels = ['positive', 'negative']
sizes = [len(test_set[test_set['Label'] == 'positive']), len(test_set[test_set['Label'] == 'negative'])]
explode = (0, 0.1)

ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax3.axis('equal')
ax3.set_title("Test Set", fontsize=14)
    
plt.show()

# %%
"""
# Model Developement
"""

# %%
"""
## Data Pre-Processing
"""

# %%
training_set.head()

# %%
"""
## Normalization
"""

# %%
training_set['Tweet'] = training_set['Tweet'].str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                                          ' ')
training_set['Tweet'] = training_set['Tweet'].str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                          ' ')
training_set['Tweet'] = training_set['Tweet'].str.replace(r'£|\$', ' ')    
training_set['Tweet'] = training_set['Tweet'].str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                                          ' ')    
training_set['Tweet'] = training_set['Tweet'].str.replace(r'\d+(\.\d+)?', ' ')

training_set['Tweet'] = training_set['Tweet'].str.replace(r'[^\w\d\s]', ' ')
training_set['Tweet'] = training_set['Tweet'].str.replace(r'\s+', ' ')
training_set['Tweet'] = training_set['Tweet'].str.replace(r'^\s+|\s+?$', '')

training_set['Tweet'] = training_set['Tweet'].str.lower()

# %%
training_set.head()

# %%
"""
### Removing Stopwords
"""

# %%
from nltk.corpus import stopwords
stop_words = nltk.corpus.stopwords.words('english')

# %%
training_set['Tweet'] = training_set['Tweet'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words))
)

# %%
training_set.head()

# %%
"""
### Lemmatization
"""

# %%
lemmatizer = nltk.stem.WordNetLemmatizer()
training_set['Tweet'] = training_set['Tweet'].apply(lambda x: ' '.join(
    lemmatizer.lemmatize(term, pos='v') for term in x.split())
)

# %%
training_set.head()

# %%
"""
### Stemming
"""

# %%
porter = nltk.PorterStemmer()
training_set['Tweet'] = training_set['Tweet'].apply(lambda x: ' '.join(
    porter.stem(term) for term in x.split())
)

# %%
training_set.head()

# %%
"""
### Tokenization
"""

# %%
training_set['Tweet'] = training_set['Tweet'].apply(lambda Tweet: nltk.word_tokenize(Tweet))

# %%
training_set.head()

# %%
"""
# Feature Extraction
"""

# %%
"""
### Vectorization
"""

# %%
corpus = training_set['Tweet'].sum()

# %%
temp_set = set(corpus)
vocabulary = list(temp_set)

# %%
len_training_set = len(training_set['Tweet'])
word_counts_per_sms = {unique_word: [0] * len_training_set for unique_word in vocabulary}

for index, sms in enumerate(training_set['Tweet']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

# %%
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()

# %%
word_counts.shape

# %%
training_set_final = pd.concat([training_set, word_counts], axis=1)
training_set_final.head()

# %%
"""
# Calculating Constants First
"""

# %%
pos_df = training_set_final[training_set_final['Label'] == 'positive'].copy()
neg_df = training_set_final[training_set_final['Label'] == 'negative'].copy()

# %%
p_pos = pos_df.shape[0] / training_set_final.shape[0]
p_neg = neg_df.shape[0] / training_set_final.shape[0]

# %%
pos_words_per_tweet = pos_df['Tweet'].apply(len)
n_pos = pos_words_per_tweet.sum()

neg_words_per_tweet = neg_df['Tweet'].apply(len)
n_neg = neg_words_per_tweet.sum()

n_vocabulary = len(vocabulary)

# %%
alpha = 1

# %%
"""
# Calculating Parameters
"""

# %%
parameters_pos = {unique_word: 0 for unique_word in vocabulary}
parameters_neg = {unique_word: 0 for unique_word in vocabulary}

for unique_word in vocabulary:
    p_unique_word_pos = (pos_df[unique_word].sum() + alpha) / (n_pos + alpha * n_vocabulary)
    p_unique_word_neg = (neg_df[unique_word].sum() + alpha) / (n_neg + alpha * n_vocabulary)
    parameters_pos[unique_word] = p_unique_word_pos
    parameters_neg[unique_word] = p_unique_word_neg

# %%
"""
# Classifying A New Tweet
"""

# %%
def tweet_classify(tweet):
    tweet = tweet.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', ' ')
    tweet = tweet.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', ' ')
    tweet = tweet.replace(r'£|\$', ' ')    
    tweet = tweet.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', ' ')    
    tweet = tweet.replace(r'\d+(\.\d+)?', ' ')

    tweet = tweet.replace(r'[^\w\d\s]', ' ')
    tweet = tweet.replace(r'\s+', ' ')
    tweet = tweet.replace(r'^\s+|\s+?$', '')
    tweet = tweet.lower()

    terms = []
    for term in tweet.split():
        if term not in set(stop_words):
            terms.append(term)
            tweet = ' '.join(terms)

    tweet = ' '.join(lemmatizer.lemmatize(term, pos='v') for term in tweet.split())            
            
    tweet = ' '.join(porter.stem(term) for term in tweet.split())  
    
    tweet = tweet.split()
    
    p_pos_given_tweet = p_pos
    p_neg_given_tweet = p_neg
    
    for word in tweet:
        if word in parameters_pos:
            p_pos_given_tweet *= parameters_pos[word]
    
        if word in parameters_neg:
            p_neg_given_tweet *= parameters_neg[word]
    
    print('P(positive|tweet):', p_pos_given_tweet)
    print('P(negative|tweet):', p_neg_given_tweet)

    if p_neg_given_tweet > p_pos_given_tweet:
        print('Label: Negative')
    elif p_neg_given_tweet < p_pos_given_tweet:
        print('Label: Positive')
    else:
        print('Equal probabilities ~ Human action needed!')

# %%
tweet_classify('TESLA SHARES EXTEND LOSS TO 4.2% ON 4Q DELIVERIES, PRICE CUTS')

# %%
"""
# Measuring the Model's Accuracy
"""

# %%
def tweet_classify_test_set(tweet):
    tweet = tweet.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', ' ')
    tweet = tweet.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', ' ')
    tweet = tweet.replace(r'£|\$', ' ')    
    tweet = tweet.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', ' ')    
    tweet = tweet.replace(r'\d+(\.\d+)?', ' ')

    tweet = tweet.replace(r'[^\w\d\s]', ' ')
    tweet = tweet.replace(r'\s+', ' ')
    tweet = tweet.replace(r'^\s+|\s+?$', '')

    tweet = tweet.lower()
    
    terms = []
    for term in tweet.split():
        if term not in set(stop_words):
            terms.append(term)
            tweet = ' '.join(terms)
    
    tweet = ' '.join(lemmatizer.lemmatize(term, pos='v') for term in tweet.split())
    
    tweet = ' '.join(porter.stem(term) for term in tweet.split())
    
    tweet = tweet.split()

    p_pos_given_tweet = p_pos
    p_neg_given_tweet = p_neg

    for word in tweet:
        if word in parameters_pos:
            p_pos_given_tweet *= parameters_pos[word]

        if word in parameters_neg:
            p_neg_given_tweet *= parameters_neg[word]

    if p_neg_given_tweet > p_pos_given_tweet:
        return 'negative' + ' ' + str(p_pos_given_tweet)
    elif p_pos_given_tweet > p_neg_given_tweet:
        return 'positive' + ' ' + str(p_pos_given_tweet)
    else:
        return 'needs human classification'

# %%
test_set['Label_predicted'] = test_set['Tweet'].apply(tweet_classify_test_set)
test_set.head()

# %%
correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['Label_predicted']:
        correct += 1

print('Results \n-------')
print('Valid:', correct)
print('Invalid:', total - correct)
print('Accuracy:', round(correct/total, 4))

# %%
def confusion_mat(y_true,y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y_pred)):
        if y_pred[i] == 'positive' and y_true[i] == 'positive':
            tp += 1
        elif y_pred[i] == 'negative' and y_true[i] == 'negative':
            tn += 1
        elif y_pred[i] == 'positive' and y_true[i] == 'negative':
            fp += 1
        elif y_pred[i] == 'negative' and y_true[i] == 'positive':
            fn += 1
    mat = np.array([tp,fn,fp,tn]).reshape(2,2)
    print("\t\tClassifier Prediction")
    print("\t\t\tPositive\tNegative")
    print("Actual | Positive\t",mat[0][0],"\t\t",mat[0][1])
    print("Value  | Negative\t",mat[1][0],"\t\t",mat[1][1])

# %%
confusion_mat(test_set['Label'], test_set['Label_predicted'])

# %%
"""
# Classifying rest of the data
"""

# %%
tweet_collection['Label_predicted'] = tweet_collection['Tweet'].apply(tweet_classify_test_set)
tweet_collection.head()

# %%
tweet_collection[['Label_pred', 'Probabilities']] = tweet_collection.Label_predicted.str.split(expand=True)
tweet_collection['D'] = tweet_collection.Date.str.split(expand=True)[0]

# %%
tweet_collection['new_prob'] = pd.to_numeric(tweet_collection['Probabilities'])
tweet_collection.head()

# %%
tsla_stock = pd.read_csv('TSLA.csv', names=['D', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], header=1)
print(tsla_stock.shape)
tsla_stock.head()

# %%
#tweet_collection.to_csv("/Users/abhinav/Desktop/final.csv")

# %%
"""
# Corelation
"""

# %%
means = tweet_collection.groupby(['D'],  as_index=False).mean()
combined = means.merge(tsla_stock, how='inner')

print(combined['new_prob'].corr(combined['Close']))

import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)

# plot the data
ax.plot(combined['new_prob'],combined['Close'], 'ro')

# %%


# %%
"""
# Logistic Regression - Own Implementation
"""

# %%
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in tweet:
            pair = (word, y)
            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair] =1
    return freqs

# %%
train_pos = training_set[training_set['Label'] == "positive"]
train_neg = training_set[training_set['Label'] == "negative"]

train_x = training_set['Tweet']
print(type(train_pos))
train_y = training_set['Label']

# %%
testing_set = test_set

# %%
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                                          ' ')
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                          ' ')
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'£|\$', ' ')    
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                                          ' ')    
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'\d+(\.\d+)?', ' ')

testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'[^\w\d\s]', ' ')
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'\s+', ' ')
testing_set['Tweet'] = testing_set['Tweet'].str.replace(r'^\s+|\s+?$', '')

testing_set['Tweet'] = testing_set['Tweet'].str.lower()

# %%
testing_set['Tweet'] = testing_set['Tweet'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words))
)

# %%
lemmatizer = nltk.stem.WordNetLemmatizer()
testing_set['Tweet'] = testing_set['Tweet'].apply(lambda x: ' '.join(
    lemmatizer.lemmatize(term, pos='v') for term in x.split())
)

# %%
porter = nltk.PorterStemmer()
testing_set['Tweet'] = testing_set['Tweet'].apply(lambda x: ' '.join(
    porter.stem(term) for term in x.split())
)

# %%
testing_set['Tweet'] = testing_set['Tweet'].apply(lambda Tweet: nltk.word_tokenize(Tweet))

# %%
test_pos = testing_set[testing_set['Label'] == "positive"]
test_neg = testing_set[testing_set['Label'] == "negative"]

test_x = testing_set['Tweet']
test_y = testing_set['Label']

# %%
train_y[train_y == 'positive'] = 1
train_y[train_y == 'negative'] = 0

train_y0 = train_y.fillna(0)
train_y1 = train_y0.to_numpy()
train_y2 = []
for i in train_y1:
    train_y2.append([i])
    
train_y3 = np.array(train_y2)

# %%
test_y[test_y == 'positive'] = 1
test_y[test_y == 'negative'] = 0

test_y0 = test_y.fillna(0)

test_y1 = test_y0.to_numpy()
test_y2 = []
for i in test_y1:
    test_y2.append([i])
    
test_y3 = np.array(test_y2)

# %%
freqs = build_freqs(train_x, train_y3)

print('Type of freqs : ', type(freqs))
print('Length of freqs : ', len(freqs))

# %%
def sigmoid(z):
    h = 1/(1+np.exp(-z))
    return h

# %%
def gradientDescent(x, y, theta, alp, iterations):
    m = x.shape[0]
    for i in range(iterations):
        z = np.dot(x, theta)
        h = sigmoid(z)

        j = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        
        theta = theta - (alp/m)* (np.dot(x.T, h-y))
        
    j = float(j)
    return j,theta

# %%
def extract_features(tweet, freqs):
    word_l = tweet
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1), 0)
        x[0, 2] += freqs.get((word, 0), 0)
    assert(x.shape==(1, 3))
    return x

# %%
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y3

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

# %%
def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs) 
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

# %%
y_hat = []
def test_logistic_regression(test_x, test_y, feqs, theta):
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred >0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    return accuracy 
  
accuracy = test_logistic_regression(test_x, test_y3, freqs, theta)
print('The accuracy of Logistic Regression is :', accuracy)

# %%
confusion_mat(test_y, y_hat)

# %%
"""
# Scikit Implementation
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

# %%
data = tweet_collection[:4997]

# %%
data['Label'] = data['Label'].map({
    'negative' : 0,
    'positive' : 1
    })

# %%
x = data["Tweet"]
y = data["Label"]

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# %%
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# %%
"""
# Logistic Regression
"""

# %%
logistic_model = LogisticRegression(solver='liblinear', penalty='l1')
logistic_model.fit(xv_train, y_train)
pred = logistic_model.predict(xv_test)
score = accuracy_score(y_test,pred)
score

# %%
cm = metrics.confusion_matrix(y_test, pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

# %%
"""
# Gradient Boosting
"""

# %%
clf1 = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0).fit(xv_train, y_train)
pred1 = clf1.predict(xv_test)
gradientboosting_score = accuracy_score(y_test,pred1)
gradientboosting_score

# %%
cm1 = metrics.confusion_matrix(y_test, pred1)
plt.figure(figsize=(9,9))
sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(gradientboosting_score)
plt.title(all_sample_title, size = 15);

# %%
"""
# AdaBoost
"""

# %%
clf2 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf2.fit(xv_train, y_train)
pred2 = clf2.predict(xv_test)
adaboost_score = accuracy_score(y_test,pred2)
adaboost_score

# %%
cm2 = metrics.confusion_matrix(y_test, pred2)
plt.figure(figsize=(9,9))
sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(adaboost_score)
plt.title(all_sample_title, size = 15);

# %%
"""
# Decision Tree
"""

# %%
def prediction(x_test, model_object):
    y_pred = model_object.predict(xv_test)
    return y_pred

# %%
res = []
for i in range(1, 10):
    model_gini = DecisionTreeClassifier(criterion = "gini",
                random_state = 123,max_depth=i, min_samples_leaf=6)

    model_gini.fit(xv_train, y_train)

    y_pred_gini = prediction(xv_test, model_gini)
    res.append(accuracy_score(y_test,y_pred_gini)*100)
print("Accuracy for each depth from 1 to 10 using Gini", res)

# %%
xpoints = range(1, 10)
plt.plot(xpoints, res)
plt.show()

# %%
res1 = []
for i in range(1, 10):
    model_entropy = DecisionTreeClassifier(
                criterion = "entropy", random_state = 123,
                max_depth = i, min_samples_leaf = 6)

    model_entropy.fit(xv_train, y_train)

    y_pred_entropy = prediction(xv_test, model_entropy)
    res1.append(accuracy_score(y_test, y_pred_entropy)*100)
print("Accuracy for each depth from 1 to 10 using Entropy", res1)

# %%
xpoints = range(1, 10)
plt.plot(xpoints, res1)
plt.show()

# %%
"""
# With Bagging and Boosting
"""

# %%
max_depth = [5, 6, 7, 8]
bag_size = [20, 40, 60, 80, 100]
terr_bag = []
terr_boo = []
for md in max_depth:
    for bs in bag_size: 
        print("Bagging : max_depth =",md,"bag_size = ",bs)
        clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state = 42, max_depth = md),n_estimators = bs)
        clf = clf.fit(xv_train, y_train)
        y_pred = clf.predict(xv_test)
        accuracy = accuracy_score(y_test,y_pred)
        print("Test Error = ", (1-accuracy)*100)
        terr_bag.append(accuracy)
        print("Confusion matrix: \n",confusion_matrix(y_test,y_pred))

max_depth = [5, 6, 7, 8]
bag_size = [20, 40, 60, 80, 100]
for md in max_depth:
    for bs in bag_size:
        print("AdaBoost : max_depth =",md,"bag_size = ",bs)
        clf = AdaBoostClassifier(tree.DecisionTreeClassifier(random_state = 42, max_depth = md),n_estimators = bs)
        clf = clf.fit(xv_train, y_train)
        y_pred = clf.predict(xv_test)
        accuracy = accuracy_score(y_test,y_pred)
        print("Test Error = ", (1-accuracy)*100)
        terr_boo.append(accuracy)
        print("Confusion matrix: \n",confusion_matrix(y_test,y_pred))

# %%
bags = np.array([20, 40, 60, 80, 100])
depth5 = terr_bag[:5]
depth6 = terr_bag[5:10]
depth7 = terr_bag[10:15]
depth8 = terr_bag[15:20]

plt.plot(bags, depth5, color='r', label='Depth 5')
plt.plot(bags, depth6, color='g', label='Depth 6')
plt.plot(bags, depth7, color='b', label='Depth 7')
plt.plot(bags, depth8, color='c', label='Depth 8')

plt.xlabel("Bag size")
plt.ylabel("Accuracy")
plt.title("Decision Tree Bagging with different Depths")
  
plt.legend()
plt.show()

# %%
bags = np.array([20, 40, 60, 80, 100])
depth5 = terr_boo[:5]
depth6 = terr_boo[5:10]
depth7 = terr_boo[10:15]
depth8 = terr_boo[15:20]

plt.plot(bags, depth5, color='r', label='Depth 5')
plt.plot(bags, depth6, color='g', label='Depth 6')
plt.plot(bags, depth7, color='b', label='Depth 7')
plt.plot(bags, depth8, color='c', label='Depth 8')

plt.xlabel("No.of Iterations")
plt.ylabel("Accuracy")
plt.title("Decision Tree Boosting with different Depths")
  
plt.legend()
plt.show()

# %%
"""
# Cross-Validation
"""

# %%
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()

sm_tree_depths = range(1,10)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(xv_train, y_train, sm_tree_depths)

plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 'Accuracy per decision tree depth on training data')    

# %%
"""
# Random Forest
"""

# %%
rfc = RandomForestClassifier(random_state=4)

rfc.fit(xv_train,y_train)

print("Train Results \n")
y_train_pred  = rfc.predict(xv_train)
y_train_prob = rfc.predict_proba(xv_train)[:,1]

print("Confusion Matrix for Train : \n", confusion_matrix(y_train, y_train_pred))
print("Accuracy Score for Train : ", accuracy_score(y_train, y_train_pred))

print("+"*50)
print("Test Results \n")
y_test_pred  = rfc.predict(xv_test)
y_test_prob = rfc.predict_proba(xv_test)[:,1]

print("Confusion Matrix for Test : \n", confusion_matrix(y_test, y_test_pred))
print("Accuracy Score for Test : ", accuracy_score(y_test, y_test_pred))

# %%
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(xv_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc,xv_test, y_test, ax=ax, alpha=0.8)
plt.show()

# %%
