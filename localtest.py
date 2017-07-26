from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.lancaster import LancasterStemmer
import nltk
import string
from sklearn.svm import LinearSVC
import json
import sys
import random
reload(sys)

sys.setdefaultencoding('utf-8')
st = LancasterStemmer()

def swear(l,s):
    for j in s:
        l = l.replace(j, j+" "+ j)
    return l

def depunc(t):
    for j in string.punctuation:
        if j != "!":
            t = t.replace(j,"")
        else:
            t = t.replace(j, " !")
    return t

with open("swear.js","r") as infile:
    sw = json.loads(infile.read())

# print len(sw)
# with open("s.txt","r") as inf:
#     for i in inf.readlines():
#         if i.strip().decode() not in sw:
#             sw.append(i.strip().decode())

print "Start"
with open("train.tsv","r") as infile:
    infile.readline()
    inlist = [[ i.split("\t")[0],i.split("\t")[1]] for i in infile.readlines()]
    trainlist = []
    testlist = []
    for line in inlist:
        if random.random()>0.2:
            trainlist.append(line)
        else:
            testlist.append(line)
    traindatalist = [swear(depunc(i[1]),sw) for i in trainlist]
    trainreslist = [i[0] for i in trainlist]
    testdatalist = [swear(depunc(i[1]),sw) for i in testlist]
    testreslist = [i[0] for i in testlist]
    print "Trainging Case: "+str(len(trainlist))
    print "Testing Case: "+str(len(testlist))

print "Finished Splitting Data"

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(class_weight= 'balanced'))
    ])

parameters = {
    'vect__ngram_range': [(1, 2)],
    'vect__max_df': [0.6],
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__C': [10]
    # 'clf__dual':(True, False)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(traindatalist, trainreslist)
# {'vect__ngram_range': (1, 2), 'clf__C': 10, 'vect__max_df': 0.6}
# print grid_search.best_params_
outres = grid_search.predict(testdatalist)

correct_count = 0
base_count = 0
for i in range(len(testlist)):
    if testreslist[i]== outres[i]:
        correct_count+=1
    if testreslist[i] == "0":
        base_count+=1

print "Local Test Accuracy: " + str(float(correct_count)/len(testlist))
print "Baseline: "+ str(float(base_count)/len(testlist))
