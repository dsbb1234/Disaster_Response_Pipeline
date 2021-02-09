from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''This class takes a string of tokens as its input and classifies each word as a noun, verb, etc.
    
    Inputs:
    BaseEstimator
    
    TransformerMixin
    
    
    
    Outputs:
    
    Summary of test results by label
    
    F1 test results
    
    Best parameters for the model
    
    '''

    def starting_verb(self, text):
        
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
