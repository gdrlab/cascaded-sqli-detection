from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from templates import Model

class Ensemble_1(Model):
  def __init__(self, pretrained_models_dict, extractors_dict ):
    self.pretrained_models_dict = pretrained_models_dict
    self.extractors_dict = extractors_dict
    super().__init__( 'ensemble_1')
    self.results = []

  def create_model(self, *args, **kwargs):
    voting_nb_tf_idf = VotingClassifier(estimators=[
      ('naive_bayes_tf-idf', self.pretrained_models_dict['naive_bayes']['tf-idf']), 
      ('naive_bayes_tf-idf_ngram', self.pretrained_models_dict['naive_bayes']['tf-idf_ngram']), 
      ('naive_bayes_bag_of_characters', self.pretrained_models_dict['naive_bayes']['bag_of_characters'])], voting='soft')
    
    self.model = voting_nb_tf_idf
    #voting_clf_bow = VotingClassifier(estimators=[('nb', naive_bayes), ('svm', svm), ('xgb', xgboost)], voting='soft')
    #voting_clf_tfidf = VotingClassifier(estimators=[('nb', naive_bayes), ('svm', svm), ('xgb', xgboost)], voting='soft')

    #self.model = MultinomialNB(*args, **kwargs)
    return 
    