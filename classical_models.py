from templates import Model
import xgboost as xgb
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

class Classical_Model(Model):
  def __init__(self, model_name, *args, **kwargs):
    super().__init__( model_name, *args, **kwargs)
    self.results = []

  def create_model(self, *args, **kwargs):
    if self.model_name == 'xgboost':
      self.model = xgb.XGBClassifier(*args, **kwargs)
    elif self.model_name == 'svm':
      self.model = svm.SVC(*args, **kwargs)
    elif self.model_name == 'naive_bayes':
      self.model = MultinomialNB(*args, **kwargs)
    else:
      raise ValueError(f"Unknown model: {self.model_name}")
    return 