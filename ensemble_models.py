from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import VotingClassifier
from templates import Model
import xgboost as xgb
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import time

# majority voting
def mv(arr):
  final_verdict = []
  for i in range(0, len(arr[0])):
    if (arr[0][i] + arr[1][i] + arr[2][i]) >= 2:
      final_verdict.append(1)
    else:
      final_verdict.append(0)
  return np.asarray(final_verdict)   

## Ensemble 1
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
    estimator_list = ['MultinomialNB', 'XGBoost', 'SVM_RBF'] # XGBoost, MultinomialNB, SVM_RBF
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
    pred_pipe1 = mv(pred_list[0:3]) 
    pred_pipe2 = mv(pred_list[3:6]) 
    pred_pipe3 = mv(pred_list[6:9]) 
    final = mv(np.asarray([pred_pipe1, pred_pipe2, pred_pipe3])) 

    return final

## Ensemble 2
class Ensemble_2(Model):
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
    super().__init__( 'ensemble_2')
    self.model = None #this will be itself

  def _create_model(self, *args, **kwargs):
    feature_ext_list = ['tf-idf', 'tf-idf_ngram', 'bag_of_characters']
    estimator_list = ['MultinomialNB', 'XGBoost', 'SVM_RBF'] # XGBoost, MultinomialNB, SVM_RBF
    
    for fe in feature_ext_list:
      for es in estimator_list:
        self.submodel.append(self.pretrained_models_dict[es][fe])
        self.subfeature_extractor.append(self.extractors_dict[fe])


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
    pred_pipe1 = mv(pred_list[0:3]) 
    pred_pipe2 = mv(pred_list[3:6]) 
    pred_pipe3 = mv(pred_list[6:9]) 
    final = mv(np.asarray([pred_pipe1, pred_pipe2, pred_pipe3])) 

    return final

## Ensemble 3
class Ensemble_3(Model):
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
    self.model = None #this will be itself
    self.combined_features = None
    self.pipeline_list = []
    super().__init__( 'ensemble_3')
    

  def _create_model(self, *args, **kwargs):
    feature_ext_list = ['tf-idf', 'tf-idf_ngram', 'bag_of_characters']
    estimator_list = ['MultinomialNB', 'XGBoost', 'SVM_RBF'] # XGBoost, MultinomialNB, SVM_RBF
    idx = 0
    
    self.submodel.append(xgb.XGBClassifier(*args, **kwargs))
    self.submodel.append(svm.SVC(*args, **kwargs))
    self.submodel.append(MultinomialNB(*args, **kwargs))
      
    for fe in feature_ext_list:
      self.subfeature_extractor.append(self.extractors_dict[fe].vectorizer)
    
    # https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html
    self.combined_features = FeatureUnion([
      ("tf-idf", self.subfeature_extractor[0]), 
      ("tf-idf_ngram", self.subfeature_extractor[1]), 
      ("bag_of_characters", self.subfeature_extractor[2])])


  def _submodel_fit(self, x_train, y_train, *args, **kwargs):
    latency = 0.0
    train_latency = 0.0
    feature_latency = 0.0

    # Use combined features to transform dataset:
    start_time = time.perf_counter()
    X_features = self.combined_features.fit(x_train, y_train).transform(x_train)
    end_time = time.perf_counter()
    self.feature_latency = (end_time - start_time)*1000/x_train.shape[0]
    #print("Combined space has", X_features.shape[1], "features")
    self.feature_size = X_features.shape[1]

    estimator_list = ['MultinomialNB', 'XGBoost', 'SVM_RBF']  # XGBoost, MultinomialNB, SVM_RBF
    self.pipeline_list.append(Pipeline([("features", self.combined_features), 
                          (estimator_list[0], self.submodel[0])])
    )
    self.pipeline_list.append(Pipeline([("features", self.combined_features), 
                          (estimator_list[1], self.submodel[1])])
    )
    self.pipeline_list.append(Pipeline([("features", self.combined_features), 
                          (estimator_list[2], self.submodel[2])])
    )
    
    start_time = time.perf_counter()
    for i in range(len(self.pipeline_list)):
      self.pipeline_list[i].fit(x_train, y_train)
    end_time = time.perf_counter()
    self.train_latency = (end_time - start_time)*1000/x_train.shape[0]

    latency = train_latency + feature_latency
    self.notes['train_time'] = latency
    return latency

  def _submodel_predict(self, x_test, *args, **kwargs):
    pred_list = []
    for i in range(len(self.pipeline_list)):
      pred_list.append(
        self.pipeline_list[i].predict(x_test)
      )
    
    pred_list = np.asarray(pred_list)
    pred_pipe_all = mv(pred_list[0:3]) 
    final = pred_pipe_all

    return final
                        

    
    