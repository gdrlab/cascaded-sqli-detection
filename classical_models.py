from templates import Model
import xgboost as xgb
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron 
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

class Classical_Model(Model):
  def __init__(self, model_name, *args, **kwargs):
    super().__init__( model_name, *args, **kwargs)
    self.results = []

  def _create_model(self, *args, **kwargs):
    if self.model_name == 'XGBoost':
      self.model = xgb.XGBClassifier(*args, **kwargs)
    elif self.model_name == 'SVM_RBF':
      self.model = svm.SVC(*args, **kwargs)
    elif self.model_name == 'SVC-GC':
      self.model = SVC(gamma=2, C=1, *args, **kwargs)
    elif self.model_name == 'NuSVC':
      self.model = NuSVC(*args, **kwargs)
    elif self.model_name == 'MultinomialNB':
      self.model = MultinomialNB(*args, **kwargs)    
    elif self.model_name == 'BernoulliNB':
      self.model = BernoulliNB(*args, **kwargs)
    elif self.model_name == 'KNeighborsClassifier':
      self.model = KNeighborsClassifier(3, *args, **kwargs)
    elif self.model_name == 'DecisionTreeClassifier':
      self.model = DecisionTreeClassifier(max_depth=5, *args, **kwargs)
    elif self.model_name == 'RandomForestClassifier':
      self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, *args, **kwargs)
    elif self.model_name == 'MLPClassifier':
      self.model = MLPClassifier(hidden_layer_sizes=(20,), alpha=1, max_iter=1000, *args, **kwargs)
    elif self.model_name == 'AdaBoostClassifier':
      self.model = AdaBoostClassifier( *args, **kwargs)
    elif self.model_name == 'BaggingClassifier':
      self.model = BaggingClassifier(*args, **kwargs)
    elif self.model_name == 'ExtraTreesClassifier':
      self.model = ExtraTreesClassifier(*args, **kwargs)
    elif self.model_name == 'LinearSVC':
      self.model = LinearSVC(*args, **kwargs)
    elif self.model_name == 'LogisticRegression':
      self.model = LogisticRegression(*args, **kwargs)
    elif self.model_name == 'NearestCentroid':
      self.model = NearestCentroid(*args, **kwargs)
    elif self.model_name == 'OneVsOneClassifier':
      clf = xgb.XGBClassifier(*args, **kwargs)
      self.model = OneVsOneClassifier(clf, *args, **kwargs)
    elif self.model_name == 'OneVsRestClassifier':
      clf = xgb.XGBClassifier(*args, **kwargs)
      self.model = OneVsRestClassifier(clf, *args, **kwargs)
    elif self.model_name == 'PassiveAggressiveClassifier':
      self.model = PassiveAggressiveClassifier(*args, **kwargs)
    elif self.model_name == 'Perceptron':
      self.model = Perceptron(*args, **kwargs)
    elif self.model_name == 'RadiusNeighborsClassifier':
      self.model = RadiusNeighborsClassifier(outlier_label=1, *args, **kwargs)
    elif self.model_name == 'RidgeClassifier':
      self.model = RidgeClassifier(*args, **kwargs)
    elif self.model_name == 'SGDClassifier':
      self.model = SGDClassifier(*args, **kwargs)
    else:
      raise ValueError(f"Unknown model: {self.model_name}")
    return
