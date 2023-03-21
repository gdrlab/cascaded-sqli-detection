from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time
from pathlib import Path
import pandas as pd

from templates import FeatureExtractor, logger
from classical_models import Classical_Model
from ensemble_models import Ensemble_1

class TestManager:
  def __init__(self, data_manager, config):
    self.data_manager = data_manager
    self.results = []
    self.config = config
    self.feature_extractors_dict = {}
    self.models_dict = {}

  def features_models_cartesian_tests(self, feature_methods, models):
    for feature_method in feature_methods:
      feature_extractor = FeatureExtractor(feature_method)
      feature_extractor.extract_features(
        self.data_manager.x_train, self.data_manager.x_test)
      self.feature_extractors_dict.update({feature_extractor.method: feature_extractor})

      for model_name in models:
        model = Classical_Model(model_name)
        model.feature_method = feature_extractor.method
        model.fit(
          feature_extractor.features['train'], self.data_manager.y_train)
        
        if model.model_name not in self.models_dict.keys():
          self.models_dict[model.model_name] = {}
        self.models_dict[model.model_name].update({feature_extractor.method: model})
        
        # Save the trained model
        timestamp = int(time.time())
        file_name = (Path(self.config['models']['dir']) 
                     / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
        model.save_model(file_name)

        y_pred = model.predict(feature_extractor.features['test'])

        accuracy = accuracy_score(self.data_manager.y_test, y_pred)
        precision = precision_score(self.data_manager.y_test, y_pred)
        recall = recall_score(self.data_manager.y_test, y_pred)
        f1 = f1_score(self.data_manager.y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.data_manager.y_test, y_pred).ravel()

        result = {
          'feature_method': feature_method,
          'model': model_name,
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
          'f1_score': f1,
          'tp': tp,
          'tn': tn,
          'fp': fp,
          'fn': fn
        }

        result.update(self.data_manager.notes)
        result.update(feature_extractor.notes)
        result.update(model.notes)
        result.update({'dataset': self.config['dataset']['path']})
        self.results.append(result)
  
  def __run_ensemble_tests(self, ensemble_models):
    for ensemble_model in ensemble_models:
      if ensemble_model == 'ensemble_1':
        self.model = Ensemble_1(self.models_dict, self.feature_extractors_dict)
      elif ensemble_model == 'ensemble_2':
        print('self.model = svm.SVC(*args, **kwargs)')
      else:
        raise ValueError(f"Unknown ensemble model: {ensemble_model}")

  def run_tests(self, feature_methods, classic_models, ensemble_models):
    self.features_models_cartesian_tests(feature_methods, classic_models)
    self.__run_ensemble_tests(ensemble_models)
    self.save_results(Path(self.config['results']['dir']))

  def save_results(self, dir):
    results_df = pd.DataFrame(self.results)
    timestamp = int(time.time())
    file_name = Path(dir) / f"results_{timestamp}.csv"
    results_df.to_csv(file_name, index=False)
    logger.info(f"Results saved to {file_name}")