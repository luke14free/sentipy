import math
import os
import pickle

def to_alphanum(s):
    import re
    pattern = re.compile('[\W_]+')
    return pattern.sub('', s)

class NaiveBayesClassifier:
    def __init__(self,*args,**kwargs):
                
        db=kwargs.get("db",{})
        categories=kwargs.get("categories",[])
        db_path=kwargs.get("db_path","/tmp/db.bin")
        persistent=kwargs.get("persistent",True)
        
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
            self.db=db
            self.db_path=db_path
            
        elif persistent and os.path.exists(db_path):
            self.db=pickle.load(open(db_path,"rb"))
            self.db_path=db_path
        
        else:
            self.db=db
            self.db_path=None
        
        self.persistent=persistent
        self.categories=categories
        
    @staticmethod
    def clean_up(tweet):
        tweet.replace("'t"," not")
        if isinstance(tweet,list):
            tweet=" ".join(tweet)
        return " ".join(filter(lambda word: len(word)>=3,map(lambda word: to_alphanum(word), tweet.split()))).lower()

    def train(self,samples,ratio={}):
        """
        Samples should have the following structure:
            [
                ('some-sample-text','positive'),
                ('some-sample-text','negative'),
                ('some-sample-text','neutral')
            ]
        Please note that for sake of speed we assume we have the same number of samples for each category.
        If not please specify it in the ratio dictionary as follows:
            {
                'positive': 2000,
                'negative': 100,
                'neutral': 123
            }
        """

        for value,category in samples:
            for word in self.clean_up(value).split():
                
                self.db.setdefault(
                              (category,word),
                                  {
                                        True:0,
                                        False:ratio.get(category,len(samples)/len(self.categories))
                                  }
                              )
                
                for other_category in filter(lambda x: x!=category,self.categories):
                    self.db.setdefault(
                                  (other_category,word),
                                    {
                                        True:0,
                                        False:ratio.get(other_category,len(samples)/len(self.categories))
                                    }
                                  )
                
                self.db[(category,word)][True]+=1
                self.db[(category,word)][False]-=1
                    
    def classify(self,sample,debug=False):
        sample=self.clean_up(sample)
                
        prob = dict([(c,0) for c in self.categories])
        gamma=0.5

        for row in self.db.keys():
            word=row[1]
            c=self.db[row][word in sample.split()]
            n=sum(self.db[row].values())
            
            #Bins will be = num of categories, unless a word recurs in every available tweet. we can safely discard that case.
            bins=len(self.categories)
            divisor = n + bins * gamma
            res=float(c + gamma) / divisor
            if debug:
                print "My naive: Word", row[1], "Sentiment", row[0], "FD",self.db[row],"C:", c,"N:", n ,"Bin:",bins,"Gamma:",gamma,"Res:",res,"Divisor:",divisor,"Log res:",math.log(res,2)
            prob[row[0]]+=math.log(res,2)
            
        inferred_label=filter(lambda k: prob[k]==max(prob.values()), prob)[0]

        return inferred_label
    
    def show_most_informative(self):
        
        def sort_function(x):
            base=float(self.db[x][True])
            reference=.0
            for category in filter(lambda c: c!= x[0], self.categories):
                reference+=self.db[(category,x[1])][True]
            return base/(reference if reference else 1)
            
            
        most_informative=sorted(self.db.keys(),
                                key=sort_function,
                                reverse=True)[:30]
                                
        for k in most_informative:
            ratio,word,sentiment=round(sort_function(k),2),k[1],k[0]
            print "Contained ratio: %s : 1 | Word: %s - Sentiment: %s" % (ratio,word,sentiment)

    def save_to_hard_disk(self,db_path=None):
        if not self.persistent:
            raise Exception("This instance of the classifier is not persistent.")

        if db_path:
            pickle.dump(self.db,open(db_path,"wb"))
        else:
            pickle.dump(self.db,open(self.db_path,"wb"))