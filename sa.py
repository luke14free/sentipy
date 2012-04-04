import nltk,urllib,json,time,naive_bayes
from functools import wraps

nltk.NaiveBayesClassifier=naive_bayes.NaiveBayesClassifier

def requires_training(function):
    @wraps(function)
    def wrap(*args):
        if not args[0].is_trained:
            raise Exception("The classifier was not trained. Call train(poitive_tweets, negative_tweets) before proceding to classification")
        return function(*args)
    return wrap

def to_alphanum(s):
    return filter(lambda x: x.isalnum(), s)

class TweetAnalyzer():
    def __init__(self):
        self.word_features = {}
        self.tweets = []
        self.is_trained = False
        self.substitutions = [
                              ("don't","do not"),
                              ("didn't","did not"),
                             ]
        
    def clean_up(self, tweet):
        if isinstance(tweet,list):
            tweet=" ".join(tweet)
        for substitution in self.substitutions:
            tweet=tweet.replace(substitution[0],substitution[1])
        return " ".join(filter(lambda word: len(word)>=3,map(lambda word: to_alphanum(word[:-1]) if word.endswith("s") else to_alphanum(word), tweet.split()))).lower()
    
    def train(self, positive_tweets, negative_tweets):
        for (tweet, sentiment) in positive_tweets + negative_tweets:
            self.tweets.append((self.clean_up(tweet).split(), sentiment))
        
        self.word_features = self.get_word_features(self.get_words_in_tweets(self.tweets))
        self.training_set = nltk.classify.util.apply_features(self.extract_features, self.tweets)
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)
        self.is_trained = True
        
    def get_words_in_tweets(self,tweets):
        all_words = []
        for words in zip(*(tweets))[0]:
            all_words.extend(words)
        return all_words
    
    def get_word_features(self,wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features
    
    def extract_features(self,document):
        if isinstance(document,basestring):
            document = document.split()
        document_words = set(self.clean_up(document).split())
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
    
    @requires_training
    def classify(self,tweet):
        tweet=filter(lambda x: x.isalnum() or x==" ",tweet)
        features=self.extract_features(tweet)
        print "Tweet: ", tweet
        print {(self.classifier.prob_classify(features).prob('positive') - self.classifier.prob_classify(features).prob('negative'))>0:'positive',
               (self.classifier.prob_classify(features).prob('positive') - self.classifier.prob_classify(features).prob('negative'))<0:'negative'}[True]
        print "Accuracy: ",abs(self.classifier.prob_classify(features).prob('positive') - self.classifier.prob_classify(features).prob('negative'))
        return self.classifier.classify(features)
    
    @requires_training
    def test_accuracy(self,test_tweets):
        """
        Expecting tweets already manually classified as:
            [('some-tweet','positive'),('some-other-tweet','negative')]
        """
        sample = map(lambda tweet: (self.extract_features(tweet[0]),tweet[1]) , test_tweets)
        
        return nltk.classify.util.accuracy(self.classifier,sample)

    @requires_training
    def show_most_informative_features(self,n=32):
        return self.classifier.show_most_informative_features(n)
        
        
"""     
pos_tweets = [
              ('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive'),
              ('I want to buy an iPad for my brother', 'positive'),
              ('I feel like buying apple shares', 'positive'),
              ('Apple shares are going up', 'positive'),
             ]

neg_tweets = [
              ('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative'),
              ("I don't like woody allen, he sucks", "negative"),
              ("I didn't like that banana split", "negative"),
              ("I would not buy an iPad as it sucks", "negative")
             ]
test_tweets= [
              ('I got in love with apple products',"positive"),
              ('I recently bought a dell computer and it sucks',"negative"),
              ('My mom says that she doesn\'t like dell, I think she\'s right',"negative"),
              ("What i love about dell is that their products are really cheap","positive"),
              ("I bet you wont catch this one fucking classifier, oh and btw I'm positive!","positive"),
              ("I will not buy an iPhone if the price remains that high", "negative"),
              ("Why should you buy an iPhone when a Galaxy Nexus is that cheaper?", "negative"),
              ("Don't believe in apple commercials, their products suck", "negative"),
              ("Once you go mac you never go back, I'm never going back to windows after trying macs", "positive")
             ]


t=TweetAnalyzer()
t.train(pos_tweets,neg_tweets)
for i in test_tweets:
    t.classify(i[0])
    print
#print t.classify("This wine is simply great!")
print "Global accuracy:", t.test_accuracy(test_tweets)*100, "%"
t.show_most_informative_features(32)



test_tweets= [
              ('I got in love with apple products',"positive"),
              ('I recently bought a dell computer and it sucks',"negative"),
              ('My mom says that she doesn\'t like dell, I think she\'s right',"negative"),
              ("What i love about dell is that their products are really cheap","positive"),
              ("I bet you wont catch this one fucking classifier, oh and btw I'm positive!","positive"),
              ("I will not buy an iPhone if the price remains that high", "negative"),
              ("Why should you buy an iPhone when a Galaxy Nexus is that cheaper?", "negative"),
              ("Don't believe in apple commercials, their products suck", "negative"),
              ("Once you go mac you never go back, I'm never going back to windows after trying macs", "positive"),
              ("not just are somebody that i used to know.. so when we said that we could not be friends", "negative"),
              ("from from from from from", "negative")
             ]
"""

try:
    #1/0
    pos_tweets=json.load(open("/tmp/deltapos.tweets","r"))
    neg_tweets=json.load(open("/tmp/deltaneg.tweets","r"))
    print "Read %s + %s tweets from hd\n" % (len(pos_tweets),len(neg_tweets))
    
except:
    pos_tweets,neg_tweets=[],[]
    error=False
    for p in range(1,201):
        print "Loading page..", p
        try:
            raw_pos_tweets,raw_neg_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":)"))).read()),json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":("))).read()) 
            print "Appending",len(raw_neg_tweets['results']),"negative tweets and",len(raw_pos_tweets['results'])
            for i in raw_pos_tweets['results']:
                pos_tweets.append((i['text'],'positive'))
            for i in raw_neg_tweets['results']:
                neg_tweets.append((i['text'],'negative'))
            time.sleep(30)
        except:
            print "Skipping page..",p
        print "Elements: %s positive, %s negative" % (len(pos_tweets),len(neg_tweets))
        
    open("/tmp/deltapos.tweets","w").write(json.dumps(pos_tweets))
    open("/tmp/de ltaneg.tweets","w").write(json.dumps(neg_tweets))

start=time.time()
t=TweetAnalyzer()
index=min(len(pos_tweets),len(neg_tweets))
zp=pos_tweets[:index]
zn=neg_tweets[:index]
print len(zp),len(zn)
t.train(zp,zn)
test_tweets=[]
z=json.loads(urllib.urlopen("http://search.twitter.com/search.json?q=%s&rpp=30&lang=en" % (urllib.quote("@Apple :)"))).read())
for m in z['results']:
    test_tweets.append(m['text'])
print "It took %s sec to start-up" % (time.time()-start)
for i in test_tweets:
    print t.classify(i)
    print
    
#print t.classify("This wine is simply great!")
#print "Global accuracy:", t.test_accuracy(test_tweets)*100, "%"
t.show_most_informative_features(32)
