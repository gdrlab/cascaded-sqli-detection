from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from templates import Model
import numpy as np

class Ensemble_1(Model):
  def __init__(self, data_mgr, pretrained_models_dict, extractors_dict ):
    self.pretrained_models_dict = pretrained_models_dict
    self.extractors_dict = extractors_dict
    self.data_manager = data_mgr
    self.results = []
    self.submodel = []
    self.subfeature_extractor = []
    self.train_latency = 0.0
    self.feature_latency = 0.0
    self.feature_size = 0
    super().__init__( 'ensemble_1')
    self.model = None #this will be itself

  def _create_model(self, *args, **kwargs):
    feature_ext_list = ['tf-idf', 'tf-idf_ngram', 'bag_of_characters']
    estimator_list = ['naive_bayes', 'xgboost', 'svm']
    idx = 0
    for es in estimator_list:
      for fe in feature_ext_list:
        self.submodel.append(self.pretrained_models_dict[es][fe])
        self.subfeature_extractor.append(self.extractors_dict[fe])
        idx = idx + 1

  def _submodel_fit(self, x_train, y_train, *args, **kwargs):
    latency = 0.0
    train_latency = 0.0
    feature_latency = 0.0
    for es in self.submodel:
      train_latency = train_latency + es.notes['train_time']
    for fe in self.subfeature_extractor:
      feature_latency = feature_latency + fe.notes['extraction_time']
      self.feature_size = self.feature_size + fe.notes['feature_size']
    
    self.train_latency = train_latency
    self.feature_latency = feature_latency

    latency = train_latency + feature_latency
    self.notes['train_time'] = latency
    return latency

  def _submodel_predict(self, x_test, *args, **kwargs):
    pred_list = []
    for i in range(0,9):
      pred_list.append(
        self.submodel[i].model.predict(
        self.subfeature_extractor[i].features['test'],
          *args, **kwargs)
      )
    
    pred_list = np.asarray(pred_list)
    pred_pipe1 = self.mv(pred_list[0:3]) 
    pred_pipe2 = self.mv(pred_list[3:6]) 
    pred_pipe3 = self.mv(pred_list[6:9]) 
    final = self.mv(np.asarray([pred_pipe1, pred_pipe2, pred_pipe3])) 

    return final

  def mv(self, arr):
    final_verdict = []
    for i in range(0, len(arr[0])):
      if (arr[0][i] + arr[1][i] + arr[2][i]) >= 2:
        final_verdict.append(1)
      else:
        final_verdict.append(0)
    return np.asarray(final_verdict)                           

    # Create pipelines
  #   naive_bayes_pipelines = self.__create_naive_bayes_pipelines()
  #   xgboost_pipelines = self.__create_xgboost_pipelines()
  #   svm_pipelines = self.__create_svm_pipelines()

  #   # Combine pipelines using VotingClassifier
  #   voting_naive_bayes = VotingClassifier(estimators=[('nb1', naive_bayes_pipelines[0]),
  #                                                     ('nb2', naive_bayes_pipelines[1]),
  #                                                     ('nb3', naive_bayes_pipelines[2])],
  #                                         voting='hard')

  #   voting_xgboost = VotingClassifier(estimators=[('xgb1', xgboost_pipelines[0]),
  #                                                 ('xgb2', xgboost_pipelines[1]),
  #                                                 ('xgb3', xgboost_pipelines[2])],
  #                                     voting='hard')
    
  #   voting_svm = VotingClassifier(estimators=[('svm1', svm_pipelines[0]),
  #                                                 ('svm2', svm_pipelines[1]),
  #                                                 ('svm3', svm_pipelines[2])],
  #                                     voting='hard')
    
  #   final_voting = VotingClassifier(estimators=[('voting_naive_bayes', voting_naive_bayes),
  #                                                 ('voting_xgboost', voting_xgboost),
  #                                                 ('voting_svm', voting_svm)],
  #                                     voting='hard')

  #   self.model = final_voting

  # def __create_naive_bayes_pipelines(self):
  #     pipeline1 = Pipeline([
  #         ('tf-idf', self.extractors_dict['tf-idf'].vectorizer),
  #         ('classifier', self.pretrained_models_dict['naive_bayes']['tf-idf'].model)
  #     ])

  #     pipeline2 = Pipeline([
  #         ('tf-idf_ngram', self.extractors_dict['tf-idf_ngram'].vectorizer),
  #         ('classifier', self.pretrained_models_dict['naive_bayes']['tf-idf_ngram'].model)
  #     ])

  #     pipeline3 = Pipeline([
  #         ('bag_of_characters', self.extractors_dict['bag_of_characters'].vectorizer),
  #         ('classifier', self.pretrained_models_dict['naive_bayes']['bag_of_characters'].model)
  #     ])

  #     return pipeline1, pipeline2, pipeline3 
  
  # def __create_xgboost_pipelines(self):
  #   pipeline1, pipeline2, pipeline3 = self.__create_naive_bayes_pipelines()

  #   pipeline1.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['tf-idf'].model
  #   pipeline2.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['tf-idf_ngram'].model
  #   pipeline3.named_steps['classifier'] = self.pretrained_models_dict['xgboost']['bag_of_characters'].model

  #   return pipeline1, pipeline2, pipeline3
  
  # def __create_svm_pipelines(self):
  #   pipeline1, pipeline2, pipeline3 = self.__create_naive_bayes_pipelines()

  #   pipeline1.named_steps['classifier'] = self.pretrained_models_dict['svm']['tf-idf'].model
  #   pipeline2.named_steps['classifier'] = self.pretrained_models_dict['svm']['tf-idf_ngram'].model
  #   pipeline3.named_steps['classifier'] = self.pretrained_models_dict['svm']['bag_of_characters'].model

  #   return pipeline1, pipeline2, pipeline3
    