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
    # Create pipelines
    naive_bayes_pipelines = self.__create_naive_bayes_pipelines()
    xgboost_pipelines = self.__create_xgboost_pipelines()
    svm_pipelines = self.__create_svm_pipelines()

    # Combine pipelines using VotingClassifier
    voting_naive_bayes = VotingClassifier(estimators=[('nb1', naive_bayes_pipelines[0]),
                                                      ('nb2', naive_bayes_pipelines[1]),
                                                      ('nb3', naive_bayes_pipelines[2])],
                                          voting='hard')

    voting_xgboost = VotingClassifier(estimators=[('xgb1', xgboost_pipelines[0]),
                                                  ('xgb2', xgboost_pipelines[1]),
                                                  ('xgb3', xgboost_pipelines[2])],
                                      voting='hard')
    
    voting_svm = VotingClassifier(estimators=[('svm1', svm_pipelines[0]),
                                                  ('svm2', svm_pipelines[1]),
                                                  ('svm3', svm_pipelines[2])],
                                      voting='hard')
    
    final_voting = VotingClassifier(estimators=[('voting_naive_bayes', voting_naive_bayes),
                                                  ('voting_xgboost', voting_xgboost),
                                                  ('voting_svm', voting_svm)],
                                      voting='hard')

    self.model = final_voting

  def __create_naive_bayes_pipelines(self):
      self.extractors_dict['tf-idf'].extract_features(self.data_manager.x_train, self.data_manager.x_test)
      self.pretrained_models_dict['naive_bayes']['tf-idf'].fit(feature_extractor.features['train'], self.data_manager.y_train)
      pipeline1 = Pipeline([
          ('tf-idf', self.extractors_dict['tf-idf']),
          ('classifier', self.pretrained_models_dict['naive_bayes']['tf-idf'])
      ])

      pipeline2 = Pipeline([
          ('tf-idf_ngram', self.extractors_dict['tf-idf_ngram']),
          ('classifier', self.pretrained_models_dict['naive_bayes']['tf-idf_ngram'])
      ])

      pipeline3 = Pipeline([
          ('bag_of_characters', self.extractors_dict['bag_of_characters']),
          ('classifier', self.pretrained_models_dict['naive_bayes']['bag_of_characters'])
      ])

      return pipeline1, pipeline2, pipeline3 
  
  def __create_xgboost_pipelines(self):
    pipeline1, pipeline2, pipeline3 = self.__create_naive_bayes_pipelines()

    pipeline1.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['tf-idf']
    pipeline2.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['tf-idf_ngram']
    pipeline3.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['bag_of_characters']

    return pipeline1, pipeline2, pipeline3
  
  def __create_svm_pipelines(self):
    pipeline1, pipeline2, pipeline3 = self.__create_naive_bayes_pipelines()

    pipeline1.named_steps['classifier'] = self.pretrained_models_dict['svm']['tf-idf']
    pipeline2.named_steps['classifier'] = self.pretrained_models_dict['svm']['tf-idf_ngram']
    pipeline3.named_steps['classifier'] = self.pretrained_models_dict['svm']['bag_of_characters']

    return pipeline1, pipeline2, pipeline3
    