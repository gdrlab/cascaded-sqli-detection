from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time
from pathlib import Path
import pandas as pd

from templates import FeatureExtractor, logger
from classical_models import Classical_Model
from ensemble_models import Ensemble_1, Ensemble_2, Ensemble_4

def evaluate(y_test, y_pred, notes):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

  result = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'tp': tp,
    'tn': tn,
    'fp': fp,
    'fn': fn
  }

  result.update(notes)
  return result


def save_results(results_dict, dest_file, *args, **kwargs):
  header=True
  for key, value in kwargs.items():
    print("{} is {}".format(key,value))
    if key == 'header':
      header=value
  
  results_df = pd.DataFrame(results_dict)
  if dest_file.is_file():
    df = pd.read_csv(dest_file)
    results_df = pd.concat([df, results_df])
    print('Appending to the existing .csv file.')

  results_df.to_csv(dest_file,  index=False, header=header) 

class TestManager:
  def __init__(self, data_manager, config, output_file_name=''):
    self.data_manager = data_manager
    self.results = []
    self.config = config
    self.feature_extractors_dict = {}
    self.models_dict = {}
    self.output_file_name = output_file_name

  def __evaluations(self, y_test, y_pred, model, feature_extractor):
    notes = {
      'feature_method': model.feature_method,
      'model': model.model_name
    }

    notes.update(self.data_manager.notes)
    notes.update(feature_extractor.notes)
    notes.update(model.notes)
    notes.update({'dataset': self.config['dataset']['file']})
    self.results.append(evaluate(y_test, y_pred, notes))
    

  def __features_models_cartesian_tests(self, feature_methods, models):
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
        if int(self.config['settings']['save_models']) != 0:
          timestamp = int(time.time())
          file_name = (Path(self.config['models']['dir']) 
                      / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
          model.save_model(file_name)

        y_pred = model.predict(feature_extractor.features['test'])
        self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
        


  def __run_ensemble_tests(self, ensemble_models):
    for ensemble_model in ensemble_models:
      model = None
      feature_extractor = None
      if ensemble_model == 'ensemble_1':
        model = Ensemble_1(self.data_manager , self.models_dict, self.feature_extractors_dict)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters'
        feature_extractor = FeatureExtractor('ensemble_1') # dummy feature ext. just for keeping latency notes.
      elif ensemble_model == 'ensemble_2':
        model = Ensemble_2(self.data_manager , self.models_dict, self.feature_extractors_dict)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters'
        feature_extractor = FeatureExtractor('ensemble_2') # dummy feature ext. just for keeping latency notes.
      elif ensemble_model == 'ensemble_4':
        model = Ensemble_4(self.data_manager , self.models_dict, self.feature_extractors_dict)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters'
        feature_extractor = FeatureExtractor('ensemble_4') # dummy feature ext. just for keeping latency notes.
      elif ensemble_model == '':
        print('No ensemble method selected in config.ini file.')
        continue
      else:
        raise ValueError(f"Unknown ensemble model: {ensemble_model}")
      
      
      # self.data_manager.x_train, self.data_manager.x_test
      model.fit(self.data_manager.x_train, self.data_manager.y_train)
      y_pred = model.predict(self.data_manager.x_test)
      feature_extractor.notes = {
          'extraction_time': model.feature_latency,
          'method': 'ensemble_1',
          'feature_size': model.feature_size,
      }
      self.model = model #this will be the saved pickle file
      self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
      # Save the trained model
      if int(self.config['settings']['save_models']) != 0:
        timestamp = int(time.time())
        file_name = (Path(self.config['models']['dir']) 
                    / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
        model.save_model(file_name) #doesn't work

      

  def __save_results(self, dir):
    if self.output_file_name == '':
      timestamp = int(time.time())
      file_name = Path(dir) / f"results_{timestamp}.csv"
    else:
      file_name = Path(dir) / self.output_file_name
    
    self.output_file_name = file_name
    save_results(self.results, dest_file=self.output_file_name, header=True)
    logger.info(f"Results saved to {self.output_file_name}")
    

  def run_tests(self, feature_methods, classic_models, ensemble_models):
    self.__features_models_cartesian_tests(feature_methods, classic_models)
    self.__run_ensemble_tests(ensemble_models)
    self.__save_results(Path(self.config['results']['dir']))