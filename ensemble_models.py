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

    
    