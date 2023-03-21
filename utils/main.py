import numpy as np
import xgboost as xgb
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
import logging
import time
import configparser
import pickle
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s l:%(lineno)d| %(message)s', 
                    datefmt='%m-%d-%Y %H:%M:%S') # type: ignore
logger = logging.getLogger(__name__)


class DataManager:
  def __init__(self, config):
    self.config = config
    self.load_data()
    self.split_data()

  def load_data(self):
    self.dataset = pd.read_csv(Path(self.config['dataset']['path']), sep='\t', engine='python')
    logger.info(f"Loaded dataset with {len(self.dataset)} rows.")

  def split_data(self):
    seed = int(self.config['data_manager']['seed'])
    split_ratio = float(self.config['data_manager']['split_ratio'])

    self.train, self.test = train_test_split(
      self.dataset, test_size=split_ratio, random_state=seed)

    self.x_train = self.train['payload'].values
    self.x_test = self.test['payload'].values
    self.y_train = self.train['label'].values
    self.y_test = self.test['label'].values

    self.notes = {
      'seed': seed,
      'split_ratio': split_ratio,
      'train_size': len(self.train),
      'test_size': len(self.test),
    }


class FeatureExtractor:
  def __init__(self, method, *args, **kwargs):
    self.method = method
    self.args = args
    self.kwargs = kwargs
    self.features = None
    self.notes = {}

  def extract_features(self, x_train, x_test):
    start_time = time.perf_counter()

    if self.method == 'tf-idf':
      vectorizer = TfidfVectorizer(*self.args, **self.kwargs)
    elif self.method == 'tf-idf_ngram':
      # Tokenizer to remove unwanted elements from out data like symbols and numbers
      token = RegexpTokenizer(r'[a-zA-Z0-9]+')
      # Using N-Gram 
      vectorizer = TfidfVectorizer(
        lowercase=True, stop_words='english', 
        ngram_range = (1, 3), # TODO: parametrize range in config file
        tokenizer = token.tokenize, analyzer='char')
    elif self.method == 'bag_of_words':
      vectorizer = CountVectorizer(analyzer='word', **self.kwargs)
    elif self.method == 'bag_of_characters':
      vectorizer = CountVectorizer(analyzer='char', **self.kwargs)
    else:
      raise ValueError(
        f"Unknown feature extraction method: {self.method}")

    self.features = {
      'train': vectorizer.fit_transform(x_train),
      'test': vectorizer.transform(x_test),
    }

    end_time = time.perf_counter()
    self.notes = {
      'extraction_time': end_time - start_time,
      'method': self.method,
      'feature_size': self.features['train'].shape[1],
    }


class Model:
  def __init__(self, model_name, *args, **kwargs):
    self.model_name = model_name
    self.model = None
    self.feature_method = ""
    self.notes = {}
    self.create_model(*args, **kwargs)

  def create_model(self, *args, **kwargs):
    if self.model_name == 'xgboost':
      self.model = xgb.XGBClassifier(*args, **kwargs)
    elif self.model_name == 'svm':
      self.model = svm.SVC(*args, **kwargs)
    elif self.model_name == 'naive_bayes':
      self.model = MultinomialNB(*args, **kwargs)
    else:
      raise ValueError(f"Unknown model: {self.model_name}")
    
  def fit(self, x_train, y_train, *args, **kwargs):
    logger.info(f"Training model: {self.model_name}")
    start_time = time.perf_counter()
    self.model.fit(x_train, y_train, *args, **kwargs)
    end_time = time.perf_counter()
    self.notes['train_time'] = end_time - start_time
    logger.info(f"Training {self.model_name} ended in: {self.notes['train_time']}sec")

  def predict(self, x_test, *args, **kwargs):
    start_time = time.perf_counter()
    y_pred = self.model.predict(x_test, *args, **kwargs)
    end_time = time.perf_counter()

    self.notes['pred_time'] = end_time - start_time
    return y_pred

  def save_model(self, file_name):
    with open(file_name, 'wb') as f:
      pickle.dump(self.model, f)
    logger.info(f"Model saved as {file_name}")

  def load_model(self, file_name, dir):
    with open(Path(dir) / file_name, 'rb') as f:
      self.model = pickle.load(f)
    logger.info(f"Model loaded from {file_name}")

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
        model = Model(model_name)
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


def main():
  config = configparser.ConfigParser()
  config.read('config.ini')
  data_manager = DataManager(config)
  test_manager = TestManager(data_manager,config=config)

  feature_methods = [method.strip() for method in config.get('feature_methods', 'methods').split(',')]
  classic_models = [model.strip() for model in config.get('models', 'classic_models').split(',')]
  ensemble_models = [model.strip() for model in config.get('models', 'ensemble_models').split(',')]

  test_manager.run_tests(feature_methods, classic_models, ensemble_models)
  


if __name__ == "__main__":
  main()
