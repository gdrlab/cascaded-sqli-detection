from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import logging
import time
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
    file_name = Path(self.config['dataset']['file'])
    file = 'datasets' / file_name
    self.dataset = pd.read_csv(file, sep='\t', engine='python')
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
    self.vectorizer = None

  def extract_features(self, x_train, x_test):
    start_time = time.perf_counter()
    # Tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    if self.method == 'tf-idf':
      self.vectorizer = TfidfVectorizer(tokenizer = token.tokenize, *self.args, **self.kwargs)
    elif self.method == 'tf-idf_ngram':
      # Using N-Gram 
      self.vectorizer = TfidfVectorizer(
        lowercase=True, stop_words='english', 
        ngram_range = (1, 3), # TODO: parametrize range in config file
        tokenizer = token.tokenize, analyzer='char')
    elif self.method == 'bag_of_words':
      self.vectorizer = CountVectorizer(analyzer='word',  **self.kwargs)
    elif self.method == 'bag_of_characters':
      self.vectorizer = CountVectorizer(analyzer='char', tokenizer = token.tokenize, **self.kwargs)
    elif self.method in ['ensemble_1', 'ensemble_2', 'ensemble_4']:
      self.vectorizer = None # created just for keeping latencies
    else:
      raise ValueError(
        f"Unknown feature extraction method: {self.method}")

    if self.vectorizer is not None:
      self.features = {
        'train': self.vectorizer.fit_transform(x_train),
        'test': self.vectorizer.transform(x_test),
      }
    else:
      self.features = {
        'train': [[0]],
        'test': [[0]],
      }
    

    end_time = time.perf_counter()
    self.notes = {
      'extraction_time': end_time - start_time,
      'method': self.method,
      'feature_size': self.features['train'].shape[1],
    }

# Generic model template. The models will be inherit this.
class Model:
  def __init__(self, model_name, *args, **kwargs):
    self.model_name = model_name
    self.model = None
    self.feature_method = ""
    self.notes = {}
    self._create_model(*args, **kwargs)

  def _create_model(self, *args, **kwargs):
    print("Parent model class create_model method. This shouldn't have been called.")
    return
  
  def _submodel_fit(self, x_train, y_train, *args, **kwargs):
    self.model.fit(x_train, y_train, *args, **kwargs)
    return 0

  def _submodel_predict(self, x_test, *args, **kwargs):
    y_pred = []
    threshold = None
    for key, value in kwargs.items():
    #print("{} is {}".format(key,value))
      if key == 'threshold':
        threshold=value
        y_pred_prob = self.model.predict_proba(x_test)[:, 1] #xgboost pred prob.
        y_pred = (y_pred_prob > threshold).astype(int)
        self.notes['threshold'] = threshold

    if threshold is None:
      y_pred = self.model.predict(x_test, *args, **kwargs)
      self.notes['threshold'] = 0.5

    return y_pred

  def fit(self, x_train, y_train, *args, **kwargs):
    logger.info(f"Training model: {self.model_name}")
    start_time = time.perf_counter()
    latency = self._submodel_fit(x_train, y_train, *args, **kwargs)
    end_time = time.perf_counter()
    if latency != 0:
      self.notes['train_time'] = latency # ensemble models calculate themselves
    else:
      self.notes['train_time'] = (end_time - start_time)*1000 / x_train.shape[0] #ms per sample
    logger.info(f"Ended training {self.model_name} in: {end_time - start_time}s")

  def predict(self, x_test, *args, **kwargs):
    start_time = time.perf_counter()
    y_pred = self._submodel_predict(x_test, *args, **kwargs)
    end_time = time.perf_counter()

    self.notes['pred_time'] = (end_time - start_time)*1000 / x_test.shape[0] #ms per sample
    return y_pred

  def save_model(self, file_name):
    with open(file_name, 'wb') as f:
      pickle.dump(self.model, f)
    logger.info(f"Model saved as {file_name}")

  def load_model(self, file_name, dir):
    with open(Path(dir) / file_name, 'rb') as f:
      self.model = pickle.load(f)
    logger.info(f"Model loaded from {file_name}")





