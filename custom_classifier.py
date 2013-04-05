import math
import os
import pickle
from stemming.porter2 import stem
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm.sparse import LinearSVC 
#from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from matplotlib.mlab import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Scaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.mlab import PCA    
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

try:
    from collections import Counter
except ImportError:
    from collections_fallback import Counter


stopwords = ['rt','re',':(',':)',':))',':((','http','im','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

def to_alphanum(s):
    import re
    return " ".join(re.findall("\w+",s))

def ngrams(tokens, MIN_N, MAX_N):
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            yield tuple(tokens[i:j])


class Classifier:
    def __init__(self,*args,**kwargs):
                
        db = kwargs.get("db",[{},{}])
        #partial_model = kwargs.get("model",) #Use NB as default model
        #model = KNeighborsClassifier()#
        model = MultinomialNB()
        #model = Pipeline([("scaler", Scaler()), ("svm", SVC())])
        
        categories = kwargs.get("categories",[])
        db_path = kwargs.get("db_path","/tmp/db.bin")
        persistent = kwargs.get("persistent",True)
        
        """
        Categories should define the possible outcomes e.g.:
            ['positive','negative','neutral']
            
        Notes:
            . if the db_path doesn't exist we'll immediately pickle db into db_path (even if it's None)

        """
        
        if persistent and not os.path.exists(db_path):
            f=open(db_path,"wb")
            pickle.dump(db,f)
            f.close()
            self.index_translator, self.db = db
            self.db_path = db_path
            
        elif persistent and os.path.exists(db_path):
            self.index_translator, self.db = pickle.load(open(db_path,"rb"))
            self.db_path = db_path
        
        else:
            self.db=db
            self.db_path=None
        
        self.persistent=persistent
        self.categories=categories
        self.model = model
        
    def clean_up(self, tweet ): 
        #Perform porter stemmer and remove any stopwords
        tweet = to_alphanum(tweet).lower()
        tweet=tweet.split(" ")            
        tweet = [word for word in tweet]
        sw = set(stopwords) #Allows for O(1) lookup
        
        #return [stem(word) for word in tweet if (word not in sw and stem(word) not in sw)]
        return ["+".join(i) for i in ngrams([stem(word) for word in tweet if (word not in sw and stem(word) not in sw)], 1, 2)]

    def train(self,samples):
        """
        Samples should have the following structure:
            [
                ('some-sample-text','positive'),
                ('some-sample-text','negative'),
                ('some-sample-text','neutral')
            ]
        """ 
        
        print "Compressing features"
                
        counter = Counter((item for sample in samples for item in self.clean_up(sample[0])))
        sorted_words = sorted([i[0] for i in counter.most_common(int(len(counter)*0.009)) if (i[0].find("+") == -1 and i[0] > 16) or (i[0].find("+") != -1 and i[1] > 6)])
        self.index_translator = dict(zip(*[sorted_words,range(len(sorted_words))]))
        
        print "Post-processed features #: %s" % len(self.index_translator)
        print "Allocating memory & word counting.."
        
        pos_counter = Counter()
        neg_counter = Counter()
        
        self.table = np.zeros((len(samples),len(self.index_translator)))
        for sample_idx in range(len(samples)):
            c = Counter(self.clean_up(samples[sample_idx][0]))
            for i in c.keys():
                if i in self.index_translator:
                    self.table[sample_idx][self.index_translator[i]] = c[i]
                    (neg_counter if samples[sample_idx][1] == 0 else pos_counter)[i] += c[i]

        
        self.pos_counter = pos_counter
        self.neg_counter = neg_counter

        classification = [sample[1] for sample in samples]
        print "Memory allocated, now training.."
        self.model.fit(self.table,classification)
        print "Trained."
        self.db = (self.index_translator, self.model)
        return self.model
    
    def score(self,test_samples):
        self.test_table = np.zeros((len(test_samples),len(self.index_translator)))
        for sample_idx in range(len(test_samples)):
            cleaned_sample_words = self.clean_up(test_samples[sample_idx][0])
            counter = Counter(cleaned_sample_words)
            for i in counter.keys():
                if i in self.index_translator:
                    self.test_table[sample_idx][self.index_translator[i]] = counter[i]
        
        classification = [sample[1] for sample in test_samples]
        return self.model.score(self.test_table,classification)
    
    def classify(self,sample):
        elem = np.zeros(len(self.index_translator))
        c = Counter(self.clean_up(sample[0]))
        for i in c.keys():
            if i in self.index_translator:
                elem[self.index_translator[i]] = c[i]
                
        try:
            prob = str(max(self.model.predict_proba(elem)[0])*100)[:5] + "%"
        except:
            prob = "n.a."
            
        return ("positive (probability: %s)" if self.model.predict(elem)[0] == 1 else "negative (probability: %s)") % (prob)
    
    def show_most_informative(self):
        #TODO: change
        
        pos_counter = self.pos_counter
        neg_counter = self.neg_counter
        
        most_common_words = list(set(pos_counter.most_common(200) + neg_counter.most_common(200)))[:100]
        raw_cross_data = np.zeros((len(most_common_words),2))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        it = 0
        for word,_ in most_common_words:
            raw_cross_data[it][0] = pos_counter[word]
            raw_cross_data[it][1] = neg_counter[word]
            it+=1
        
        pca_results = PCA(raw_cross_data)
        
        results = pca_results.Y
        af = AffinityPropagation().fit(results)


        
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.,1.)


        it = 0
        for word,_ in most_common_words:
            ax.text(s=word, x = results[it][0], y = results[it][1])
            it+=1
            
        from itertools import cycle
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        X = results
        n_clusters_ = len(cluster_centers_indices)

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X[cluster_centers_indices[k]]
            pl.plot(X[class_members, 0], X[class_members, 1], col + '.')
            pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=14)
            for x in X[class_members]:
                pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    
        
        print "Variance explained by the PCA components: %s " % pca_results.fracs
        plt.show()

    def save_to_hard_disk(self,db_path=None):
        #Not implemented yet.
        pass