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

with open("train.tsv","r") as trainfile:
    trainfile.readline()
    trainlist = [[ i.split("\t")[0],i.split("\t")[1]] for i in trainfile.readlines()]
    traindatalist = [swear(depunc(i[1]),sw) for i in trainlist]
    trainreslist = [i[0] for i in trainlist]

print "Finished Loading Training Set"

with open("test.tsv","r") as testfile:
    testfile.readline()
    testlist = [[ i.split("\t")[0],i.split("\t")[1]] for i in testfile.readlines()]
    testdatalist = [swear(depunc(i[1]),sw) for i in testlist]
    testIDlist = [i[0] for i in testlist]

print "Finished Loading Testing Set"

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


with open("submission.csv","w") as fout:
    fout.write("Id,Category\n")
    for i in testIDlist:
        fout.write(i)
        fout.write(",")
        fout.write(outres[int(i)])
        fout.write("\n")

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
