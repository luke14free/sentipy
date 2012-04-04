import math


def to_alphanum(s):
    import re
    pattern = re.compile('[\W_]+')
    return pattern.sub('', s)

class MyNaiveBayesClassifier:
    def __init__(self,db=None,categories=None):
        self.db=db
        self.categories=categories
    
    @staticmethod
    def clean_up(tweet):
        tweet.replace("'t"," not")
        if isinstance(tweet,list):
            tweet=" ".join(tweet)
        return " ".join(filter(lambda word: len(word)>=3,map(lambda word: to_alphanum(word), tweet.split()))).lower()

    def train(self,samples,categories,ratio=0.5):
        """
        Samples should have the following structure:
            [
                ('some-sample-text','positive'),
                ('some-sample-text','negative'),
                ('some-sample-text','neutral')
            ]
            
        Categories should define the possible outcomes:
            ['positive','negative','neutral']
        """
        db={}
        
        words = list(set(reduce(lambda x,y: x+self.clean_up(y[0]).split(),samples,[])))

        for value,category in samples:
            for word in self.clean_up(value).split():
                db[(category,word)]=db.get((category,word),{True:0,False:len(samples)/len(categories)}) #Assuming we have the same number of samples for each cat.
                for cat in filter(lambda c: c!=category,categories):
                    db[(cat,word)]=db.get((cat,word),{True:0,False:len(samples)/len(categories)})
                db[(category,word)][True]+=1
                db[(category,word)][False]-=1
     
        
        #Ideally: create an immutable instance of discodb with the database in it.
        #print db
        return MyNaiveBayesClassifier(db,categories)
        
    def classify(self,sample):
        #print self.db
        sample=self.clean_up(sample)
        
        #prob={c:0 for c in self.categories} Not cython compliant..! :P
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
            #print "My naive: Word", row[1], "Sentiment", row[0], "FD",self.db[row],"C:", c,"N:", n ,"Bin:",bins,"Gamma:",gamma,"Res:",res,"Divisor:",divisor,"Log res:",math.log(res,2)
            prob[row[0]]+=math.log(res,2)
            #print "My naive: Word", row[1], "Sentiment", row[0], "FD",self.db[row],"C:", c,"N:", n ,"Bin:",bins,"Gamma:",gamma,"Res:",res,"Divisor:",divisor,"Log res:",math.log(res,2)
            
        l1,p1=filter(lambda k: prob[k]==max(prob.values()), prob)[0],max(prob.values())
        del prob[l1]
        l2,p2=filter(lambda k: prob[k]==max(prob.values()), prob)[0],max(prob.values())
        accuracy = float(p1)/p2
        return l1,accuracy