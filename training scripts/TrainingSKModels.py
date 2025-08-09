import praw
import pandas as pd
import re
import string
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score, confusion_matrix

"""
Train set of SKLearn models, using Grid to run all parameters over each model
"""

class TrainSKModels:
    def __init__(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data["comment"], data["sentiment"], test_size=0.2, random_state=65, shuffle=True, stratify=data["sentiment"])
        self.X_train = [str(i) for i in self.X_train]
        self.X_test = [str(i) for i in self.X_test]
        
        print(self.y_test.value_counts(normalize=True))
        self.y_train = [int(i) for i in self.y_train]
        self.y_test = [int(i) for i in self.y_test]
        self.stop_words = self.process_stop_words()
        
    def process_stop_words(self):
        stop_words = stopwords.words('english')
        stop_words.extend(additional_politics_english_stop)
        stop_words.extend(additional_english_stop)
        new_stop_list = stop_words
        pos_freq = nltk.FreqDist(new_stop_list)
        most_common_right = pos_freq.most_common(1500)

    def train_all_models(self):
        self.cvec_xgb_train()
        self.cvec_lr_train()
        self.cvec_tfid_clf_train()

    def cvec_xgb_train(self):
        pipe = Pipeline([
        ('cvec', CountVectorizer()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error'))
        ])
        
        pipe_params = {
        'cvec__max_features': [None,750,1500],
        'cvec__min_df': [3,5],
        'cvec__max_df': [.4,.3],
        'cvec__ngram_range': [(1,2),(2,3),(1,3)],
        'cvec__stop_words': [None, 'english', self.stop_words]
        }
        
        self.perform_training(pipe, pipe_params, "cvec_xgb")
        
    def cvec_lr_train(self):
        pipe = Pipeline([
        ('cvec', CountVectorizer()),
        ('lr', LogisticRegression())
        ])

        pipe_params = {
        'cvec__max_features': [None,500,1000,1500],
        'cvec__min_df': [2,3],
        'cvec__max_df': [.3,.4,],
        'cvec__ngram_range': [(1,2),(1,3)],
        'cvec__stop_words': [None,'english', self.stop_words],
        'lr__penalty': ['l2'],
        'lr__max_iter': [5000, 25000]
        }
        
        self.perform_training(pipe, pipe_params, "cvec_lr")
        
    def cvec_tfid_clf_train(self):
        pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
        ])
        
        pipe_params = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__max_iter': (10, 50, 80),
        }
        
        self.perform_training(pipe, pipe_params, "cvec_tfid_clf")
        
        
    def perform_training(self, pipe, pipe_params, name):
        gs = GridSearchCV(pipe, param_grid=pipe_params, cv = 5, verbose = 1, n_jobs = -1)
    
        gs.fit(self.X_train,self.y_train)
        
        cvxgb_bestscore = gs.best_score_
        cvxgb_params = gs.best_params_
        cvxgb_train = gs.score(self.X_train, self.y_train)
        cvxgb_test= gs.score(self.X_test, self.y_test)
        cvxgb = ('CountVec with XGBoost',cvxgb_bestscore, cvxgb_params, cvxgb_train, cvxgb_test)
        
        print(f'Best Score: {gs.best_score_}')
        print(f'Best Parameters: {gs.best_params_}')
        print(f'Train Accuracy Score: {gs.score(self.X_train, self.y_train)}')
        print(f'Test Accuracy Score: {gs.score(self.X_test,self.y_test)}')
            
        from joblib import dump, load
        dump(gs, name+'-model.joblib')