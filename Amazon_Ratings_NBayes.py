import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, auc, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

########### Importing data and displaying first 10 records ###########
data = pd.read_csv('amazon_alexa.tsv',delimiter='\t')
print(data.head(10))

############# Shape and description of the data ############
print("Shape:", data.shape)
print(data.describe())

############## Visualization #################
plt.figure(figsize=(12, 7))
sns.scatterplot(x="rating", y="rating", hue="feedback",data=data)
plt.title("Relation between Rating and Overall Feedback");

""" the above plot shows for feedback to be positive (1), rating >= 3"""

fig, axs = plt.subplots(1, 2, figsize=(24, 10))

data.feedback.value_counts().plot.barh(ax=axs[0])
axs[0].set_title(("Class Distribution - Feedback {1 (positive) & 0 (negative)}"));

data.rating.value_counts().plot.barh(ax=axs[1])
axs[1].set_title("Class Distribution - Ratings");

'''
We have a highly positive skewed distribution here in both cases, 
and these products have been pretty well received by the customers !'''

data.variation.value_counts().plot.barh(figsize=(12, 7))
plt.title("Class Distribution - Variation");

#We have distinct bins here, showing pattern of customer preference for various models, with Black Dot as the most 
#popular one

############# Variation wise Mean Ratings ###############

data.groupby('variation').mean()[['rating']].plot.barh(figsize=(12, 7))
plt.title("Variation wise Mean Ratings");

"""
No obvious patterns here, all the variations of the 
product have been equally well received
"""

data['review_length'] = data.verified_reviews.str.len()

pd.DataFrame(data.review_length.describe())

########## Histogram of review length ##########

data['review_length'].plot.hist(bins=200, figsize=(16, 7))
plt.title("Histogram of Review Lengths");

data.groupby('rating').mean()[['review_length']].plot.barh(figsize=(12, 7))
plt.title("Mean Length of Reviews - Grouped by Ratings");

"""
Rating 2 : Customers tend to describe the flaws in detail 
and it's natural to be vocal about something you didn't find good.

Rating 5 : There could broadly be two kinds of reviews here; 
people who actually describe the positives in about 100 words or so, and the ones who 
comment "Awesome", "Loved it !" etc.

"""
########################## Most frequent words  #####################

cv = CountVectorizer(stop_words='english')
cv.fit_transform(data.verified_reviews);
vector = cv.fit_transform(data.verified_reviews)
sum_words = vector.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
freq_df = pd.DataFrame(words_freq, columns=['word', 'freq'])
freq_df.head(15).plot(x='word', y='freq', kind='barh', figsize=(20, 12))
plt.title("Most Frequently Occuring Words - Top 15");

""" from the chart we can see that the word love is used 
highest times and amazon is used the lowest times """

################## Word cloud ########

wordcloud = WordCloud(background_color='white',width=800, height=500).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22);

# from this word cloud we see the highest and lowest word visually.

########## Dropping date and variation columns ###########
data = data.drop(['date', 'variation'], axis=1)
print(data.head(5))

#Checking the class labels distribution
classes = data['feedback']
print(classes.value_counts(),"\n")

ratings = data['rating']
print(ratings.value_counts())

#Using regular expressions to remove URLs, numbers etc
processed = data['verified_reviews'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',' ')
processed = processed.str.replace(r'http',' ')
processed = processed.str.replace(r'£|\$', ' ')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' ')
processed = processed.str.replace(r'\d+(\.\d+)?', ' ')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', ' ')
processed = processed.str.replace(r'\d+',' ')
processed = processed.str.lower()

############### Removing stop words from reviews ##################
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

nltk.download('punkt')

#Creating bag-of-words
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

############ Printing the total number of words and the 15 most common words ##############
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))

############# Using the 1st 1000 words as features #################
word_features = list(all_words)[:1000]

#The find_features function will determine which of the 1000 word features are contained in the reviews
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

#Unifying reviews with their respective class labels
messages = list(zip(processed, data['feedback']))
print("Testing(Unification of review, feedback, and rating):", messages[0])

#Defining a seed for reproducibility and shuffling
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#Forming a featureset from reviews and labels
featuresets = [(find_features(text), label) for (text, label) in messages]
print(featuresets[0])

#Splitting the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.3, random_state=1)
print("Training set:", len(training))
print("Testing set:", len(testing))

#Defining Naive Bayes model for training
model = MultinomialNB()
#Training the model on the training data and calculating accuracy
nltk_model = SklearnClassifier(model)
nltk_model.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Naive Bayes Accuracy: {}".format(accuracy))

#Listing the predicted labels for testing dataset and computing error value
txt_features, labels = list(zip(*testing))
prediction = nltk_model.classify_many(txt_features)
print("Mean Absoulte Error:", mean_absolute_error(prediction, labels) *100)

#ROC and AUC
fpr, tpr, thresholds = roc_curve(labels, prediction)
roc_auc = auc(fpr,tpr)
roc_auc

#ROC plot
plt.plot(fpr,tpr,color='darkorange',label='ROC Curve (area=%0.2f)' % roc_auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

# Bag of words - Negative/Positive reviews w.r.t ratings
neg_1=[];neg_2=[];pos_2=[];pos_3=[];pos_4=[];pos_5=[];
unifyy = list(zip(prediction, txt_features, labels, data['rating']))
for p, t, l, r in unifyy:
    for key, value in t.items():
        if value==True and l==p==0 and r==1:
            neg_1.append(key)
            break
        elif value==True and l==p==0 and r==2:
            neg_2.append(key)
            break
        elif value==True and l==p==1 and r==2:
            pos_2.append(key)
            break
        elif value==True and l==p==1 and r==3:
            pos_3.append(key)
            break
        elif value==True and l==p==1 and r==4:
            pos_4.append(key)
            break
        elif value==True and l==p==1 and r==5:
            pos_5.append(key)
            break
print("Negative review words with 1 rating:", np.unique(neg_1))
print("Negative review words with 2 rating:", np.unique(neg_2))
print("Positive review words with 2 rating:", np.unique(pos_2))
print("Positive review words with 3 rating:", np.unique(pos_3))
print("Positive review words with 4 rating:", np.unique(pos_4))
print("Positive review words with 5 rating:", np.unique(pos_5))

#Printing a confusion matrix and classification report
print(classification_report(labels, prediction))

#Displaying the false positives and True positives in confusion matrix
df = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0','1']])
print(df)

#Generating wordcloud for reviews
if(len(neg_1) > 0):
    negtv_1 = str(neg_1)
    wordCloud = WordCloud(background_color="white").generate(negtv_1)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative reviews with 1 rating Words')
    plt.show()

if(len(neg_2) > 0):
    negtv_2 = str(neg_2)
    wordCloud = WordCloud(background_color="white").generate(negtv_2)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative reviews with 2 rating Words')
    plt.show()

if(len(pos_2) > 0):
    postv_2 = str(pos_2)
    wordCloud = WordCloud(background_color="white").generate(postv_2)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 2 rating Words')
    plt.show()

if(len(pos_3) > 0):
    postv_3 = str(pos_3)
    wordCloud = WordCloud(background_color="white").generate(postv_3)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 3 rating Words')
    plt.show()

if(len(pos_4) > 0):
    postv_4 = str(pos_4)
    wordCloud = WordCloud(background_color="white").generate(postv_4)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 4 rating Words')
    plt.show()

if(len(pos_5) > 0):
    postv_5 = str(pos_5)
    wordCloud = WordCloud(background_color="white").generate(postv_5)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 5 rating Words')
    plt.show()