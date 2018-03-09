
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from sklearn.svm import LinearSVC
import json
import sys
import random
reload(sys)

sys.setdefaultencoding('utf-8')


def swear(l,s):
    # manually put more weight on those swear words
    for j in s:
        l = l.replace(j, j+" "+ j)
    return l

def depunc(t):
    for j in string.punctuation:
        # only keep exclamation marks
        if j != "!":
            t = t.replace(j,"")
        else:
            t = t.replace(j, " !")
    return t

def main():
    with open("swear.js","r") as infile:
        sw = json.loads(infile.read())

    print "Start"
    with open("train.tsv","r") as infile:
        infile.readline()
        inlist = [[ i.split("\t")[0],i.split("\t")[1]] for i in infile.readlines()]

        random.shuffle(inlist)

        trainlist = [v for i,v in enumerate(inlist) if i%5 != 0]
        cvlist = [v for i,v in enumerate(inlist) if i%5 == 0]

        traindatalist = [swear(depunc(i[1]),sw) for i in trainlist]
        trainreslist = [i[0] for i in trainlist]
        cvdatalist = [swear(depunc(i[1]),sw) for i in cvlist]
        cvreslist = [i[0] for i in cvlist]
        print "Trainging Case: "+str(len(trainlist))
        print "Testing Case: "+str(len(cvlist))

    print "Finished Splitting Data"

    print "Start fitting model"
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LinearSVC(class_weight= 'balanced'))
        ])

    parameters = {
        'vect__ngram_range': [(1, 2)],
        'vect__max_df': [0.5,1],
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__C': [0,5, 1, 5, 10]
        # 'clf__dual':(True, False)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(traindatalist, trainreslist)
    print "Selected Parameters:"
    print grid_search.best_params_
    outres = grid_search.predict(cvdatalist)
    print grid_search.score(cvdatalist, cvreslist)

    print "Local Test Accuracy: " + str(float(correct_count)/len(testlist))

    print "Baseline: {0}".format(float(len([i for i in cvreslist if i == "0"]))/len(cvlist))

main()
