import json
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from templates import FeatureExtractor, logger
from classical_models import Classical_Model
from ensemble_models import Ensemble_1, Ensemble_2, Ensemble_3
from tqdm import tqdm
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
    logger.info("{} is {}".format(key,value))
    if key == 'header':
      header=value
  
  results_df = pd.DataFrame(results_dict)
  if dest_file.is_file():
    df = pd.read_csv(dest_file)
    results_df = pd.concat([df, results_df])
    logger.info('Appending to the existing .csv file.')

  results_df.to_csv(dest_file,  index=False, header=header) 

class TestManager:
  def __init__(self, data_manager, config, output_file_name=''):
    self.data_manager = data_manager
    self.results = []
    self.all_params = []
    self.config = config
    self.feature_extractors_dict = {}
    self.feature_extractors_dict_for_ensemble = {}
    self.models_dict = {}
    self.models_dict_for_ensemble = {}
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
    total_feature_extractors = len(feature_methods)
    total_models = len(models)
    idx_fe = 1
    
    for feature_method in tqdm(feature_methods, desc='Feature extractors'):
      logger.info(f'running {idx_fe} of {total_feature_extractors} feature extractors.')
      idx_fe += 1
      feature_extractor = FeatureExtractor(feature_method)
      feature_extractor.extract_features(
        self.data_manager.x_train, self.data_manager.x_test)
      self.feature_extractors_dict.update({feature_extractor.method: feature_extractor})
      idx_model = 1
      for model_name in tqdm(models, desc='Models'):
        logger.info(f'running {idx_model} of {total_models} models.')
        idx_model += 1
        model = Classical_Model(model_name)
        model.feature_method = feature_extractor.method
        model.fit(
          feature_extractor.features['train'], self.data_manager.y_train)
        model.vectorizer = feature_extractor.vectorizer 
        
        if model.model_name not in self.models_dict.keys():
          self.models_dict[model.model_name] = {}
        self.models_dict[model.model_name].update({feature_extractor.method: model})
        
        # Save the trained model
        if int(self.config['settings']['save_models']) != 0:
          # save model only if it was not saved before:
          if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
            timestamp = int(time.time())
            file_name = (Path(self.config['models']['dir']) 
                        / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
            model.save_model(file_name)

        y_pred = model.predict(feature_extractor.features['test'])
        self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
        params = model.model.get_params()
        #Problem: XGboost as a parameter of OneVsRest, cannot be serialized with json.dump
        #dirty fix: just save the name, not the object
        if 'estimator' in params:
          params['estimator'] = type(params['estimator']).__name__
        
        # save parameters only if it was not saved before:
        if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
          self.__append_model_params_file(model_name=model_name, fe=feature_extractor.method, model_params=params)

  def __features_models_cartesian_for_ensembles(self, feature_methods, models):
    total_feature_extractors = len(feature_methods)
    total_models = len(models)
    idx_fe = 1
    
    for feature_method in tqdm(feature_methods, desc='Feature extractors'):
      logger.info(f'running {idx_fe} of {total_feature_extractors} feature extractors.')
      idx_fe += 1
      feature_extractor = FeatureExtractor(feature_method)
      feature_extractor.extract_features(
        self.data_manager.x_train, self.data_manager.x_test)
      self.feature_extractors_dict_for_ensemble.update({feature_extractor.method: feature_extractor})
      idx_model = 1
      for model_name in tqdm(models, desc='Models'):
        logger.info(f'running {idx_model} of {total_models} models.')
        idx_model += 1
        model = Classical_Model(model_name)
        model.feature_method = feature_extractor.method
        model.fit(
          feature_extractor.features['train'], self.data_manager.y_train)
        model.vectorizer = feature_extractor.vectorizer 
        
        if model.model_name not in self.models_dict_for_ensemble.keys():
          self.models_dict_for_ensemble[model.model_name] = {}
        self.models_dict_for_ensemble[model.model_name].update({feature_extractor.method: model})
        
        # Save the trained model
        # if int(self.config['settings']['save_models']) != 0:
        #   # save model only if it was not saved before:
        #   if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
        #     timestamp = int(time.time())
        #     file_name = (Path(self.config['models']['dir']) 
        #                 / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
        #     model.save_model(file_name)

        y_pred = model.predict(feature_extractor.features['test'])
        self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
        params = model.model.get_params()
        #Problem: XGboost as a parameter of OneVsRest, cannot be serialized with json.dump
        #dirty fix: just save the name, not the object
        if 'estimator' in params:
          params['estimator'] = type(params['estimator']).__name__
        
        # save parameters only if it was not saved before:
        if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
          self.__append_model_params_file(model_name=model_name, fe=feature_extractor.method, model_params=params)
          
  def __append_model_params_file(self, model_name, fe, model_params):
    self.all_params.append((model_name, fe, model_params))
  
  def __save_all_params_file(self, dir):
    if self.output_file_name == '':
      currentDateAndTime = datetime.now()
      currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
      file_name = Path(dir) / f'results_{currentTime}.csv'
    else:
      file_name = Path(dir) / Path(self.output_file_name).name
    
    self.output_file_name = file_name
    param_file_name = file_name.parent / file_name.with_suffix('.json').name.replace('results', 'params')
    
    # Save the parameters to a file
    with open(param_file_name, 'w') as f:
      json.dump(self.all_params, f)
    logger.info(f"Params saved to {param_file_name}")

  def __run_ensemble_tests(self, ensemble_models):
    feature_methods =  ['tf-idf', 'tf-idf_ngram', 'bag_of_characters']
    classic_models = ['MultinomialNB', 'XGBoost', 'SVM_RBF']
    self.__features_models_cartesian_for_ensembles(feature_methods, classic_models)
    for ensemble_model in ensemble_models:
      model = None
      feature_extractor = None
      if ensemble_model == 'ensemble_1':
        model = Ensemble_1(self.data_manager , self.models_dict_for_ensemble, self.feature_extractors_dict_for_ensemble)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters' # XGBoost, MultinomialNB, SVM_RBF
        feature_extractor = FeatureExtractor('ensemble_1') # dummy feature ext. just for keeping latency notes.
      elif ensemble_model == 'ensemble_2':
        model = Ensemble_2(self.data_manager , self.models_dict_for_ensemble, self.feature_extractors_dict_for_ensemble)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters'
        feature_extractor = FeatureExtractor('ensemble_2') # dummy feature ext. just for keeping latency notes.
      elif ensemble_model == 'ensemble_3':
        model = Ensemble_3(self.data_manager , self.models_dict_for_ensemble, self.feature_extractors_dict_for_ensemble)
        model.feature_method = 'tf-idf, tf-idf_ngram, bag_of_characters'
        feature_extractor = FeatureExtractor('ensemble_3') # dummy feature ext. just for keeping latency notes.
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

  def __adaptive(self, selected_feature_method, selected_model_name, threshold=0.5):
   
    feature_extractor = FeatureExtractor(selected_feature_method)
    feature_extractor.extract_features(
      self.data_manager.x_train, self.data_manager.x_test)
    self.feature_extractors_dict.update({feature_extractor.method: feature_extractor})

 
    model = Classical_Model(selected_model_name)
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

    y_pred = model.predict(feature_extractor.features['test'], threshold=threshold)
    self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
    return model      

  def __save_results(self, dir):
    if self.output_file_name == '':
      currentDateAndTime = datetime.now()
      currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
      file_name = Path(dir) / f'results_{currentTime}.csv'
    else:
      file_name = Path(dir) / Path(self.output_file_name).name
    
    self.output_file_name = file_name
    save_results(self.results, dest_file=self.output_file_name, header=True)
    logger.info(f"Results saved to {self.output_file_name}")

  def __plot_roc(self, file_name):
    # predict probabilities of positive class for the test set
    y_test, y_pred_prob = self.__load_pred_prob(file_name)
    #y_pred = (y_pred_prob > 0.001).astype(int)
    #tn, fp, fn, tp = confusion_matrix(self.data_manager.y_test, y_pred).ravel()
    #print(tn, fp, fn, tp) # 2287 1562 1 2272

    # calculate the false positive rate and true positive rate for different thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
  
  def __plot_roc2(self, file_name):
    # predict probabilities of positive class for the test set
    y_test, y_pred_prob = self.__load_pred_prob(file_name)


    # set the threshold range and step size
    threshold_range = np.arange(0, 0.1, 0.001)

    # calculate the false negative rate for each threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    fnr = 1 - tpr

    # select the false negative rates for the given threshold range
    fnr_range = [fnr[np.argmin(np.abs(thresholds - t))] for t in threshold_range]

    # plot the ROC curve for false negatives
    plt.plot(threshold_range, fnr_range, label='False Negatives')

    # plot the reference line at 50% false negatives
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='gray')

    # plot the focus area with a red rectangle
    plt.fill_between(threshold_range, fnr_range, where=(threshold_range >= 0) & (threshold_range <= 0.1), alpha=0.2, color='red')

    # set the plot title and axis labels
    plt.title('ROC Curve for False Negatives')
    plt.xlabel('Threshold')
    plt.ylabel('False Negative Rate')

    # set the plot limits and legend
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()

    # show the plot
    plt.show()

  def __plot_roc3(self, file_name):
    # predict probabilities of positive class for the test set
    y_test, y_pred_prob = self.__load_pred_prob(file_name)


    # Define the threshold range
    threshold_range = np.arange(0, 0.1, 0.001)

    # Calculate the confusion matrix for each threshold value
    fp_rate_list = []
    fn_rate_list = []
    for threshold in threshold_range:
      y_pred_binary = (y_pred_prob >= threshold).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
      fp_rate = fp / (fp + tn)
      fn_rate = fn / (fn + tp)
      fp_rate_list.append(fp_rate)
      fn_rate_list.append(fn_rate)

    # Plot the FPR vs FNR curve
    plt.plot(fp_rate_list, fn_rate_list, label='FPR vs FNR', color='red')

    # Add the point where FPR=0 and FNR=0 as a circle marker
    plt.plot(0, 0, 'o', markersize=8, color='green')

    # Add a title, axis labels, and legend
    plt.title('False Positive Rate vs False Negative Rate')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.legend(loc='best')

    # Set the limits of the plot
    plt.xlim(0, 0.06)
    plt.ylim(0, 0.2)

    # Show the plot
    plt.show()
  
  def __save_pred_prob(self, adaptive_model, dir=""):
    y_pred_prob = adaptive_model.model.predict_proba(
      self.feature_extractors_dict['tf-idf_ngram'].features['test'])[:, 1]
    #result = (y_pred_prob, self.data_manager.y_test)
    y_test = self.data_manager.y_test
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
    file_name = Path(dir) / f'xgboost_pred_{currentTime}.csv'
    df = pd.DataFrame({'y_pred_prob': y_pred_prob, 'y_test': y_test})
    df.to_csv(file_name, index=False)
    print(f'Prediction probabilities are saved to {file_name}')
    return file_name

  def __load_pred_prob(self, file_name):
    df = pd.read_csv(file_name)
    y_pred_prob = df['y_pred_prob'].values
    y_test = df['y_test'].values
    return y_test, y_pred_prob

  # Train and evaluate the first stage
  def __first_stage(self):
    pass_aggressive_threshold = -0.3 #prediction
    #scale_pos_weight = 5000.0
    class_weights = {1: 0.999, 0: 0.001}
    #class_weights = None

    feature_method = 'tf-idf_ngram'
    #model_name = 'XGBoost'
    model_name = 'PassiveAggressiveClassifier'

    feature_extractor = FeatureExtractor(feature_method)
    start_time = time.time()
    feature_extractor.extract_features(
        self.data_manager.x_train, self.data_manager.x_test)
    stop_time = time.time()
    extraction_time = ((stop_time - start_time)*1000 
                      / (len(self.data_manager.x_train) + len(self.data_manager.x_test)) )

    #model = Classical_Model(model_name, scale_pos_weight=scale_pos_weight)
    model = Classical_Model(model_name, class_weight=class_weights)
    model.feature_method = feature_extractor.method
    start_time = time.time()
    model.fit(
        feature_extractor.features['train'], 
        self.data_manager.y_train) #scale_pos_weight=5.0
    stop_time = time.time()
    training_time = (stop_time - start_time)*1000 / feature_extractor.features['train'].shape[0]


    start_time = time.time()
    first_stage_y_pred = model.predict(feature_extractor.features['test'], pass_aggressive_threshold=pass_aggressive_threshold)
    stop_time = time.time()

    testing_time = (stop_time - start_time)*1000 / feature_extractor.features['test'].shape[0]


    # Save results to csv file
    notes={}
    result = evaluate(self.data_manager.y_test, first_stage_y_pred, notes=notes)
    print(result)
    #save_results([result], proposed_test_results_file)

    # Extract the positive predicitions for the second stage
    first_stage_positive_preds = self.data_manager.x_test[(first_stage_y_pred == 1)]
    first_stage_positive_preds_true_labels = self.data_manager.y_test[(first_stage_y_pred == 1)]
    fs_pos_len = len(first_stage_positive_preds)
    nof_test_samples = len(self.data_manager.x_test)
    fs_rat = fs_pos_len/nof_test_samples
    print(f"Positive predicitions in the first stage: {fs_pos_len} out of {nof_test_samples}. Ratio:{fs_rat}")

    return first_stage_positive_preds, first_stage_positive_preds_true_labels
  
  def run_first_stage_tests(self):
    first_stage_y_pred, first_stage_positive_preds_true_labels = self.__first_stage()


  def run_tests(self, feature_methods, classic_models, ensemble_models):
    self.__features_models_cartesian_tests(feature_methods, classic_models)
    self.__run_ensemble_tests(ensemble_models)
    
    #self.__adaptive('tf-idf_ngram', 'xgboost', threshold=0.5)
    #adaptive_model = self.__adaptive('tf-idf_ngram', 'xgboost', threshold=0.3)
    #file_name = self.__save_pred_prob(adaptive_model,dir=Path(self.config['results']['dir']))
    # TODO: save pred_prob and open it in display results ipython file.
    # TODO: plot ROC for 0.0001 values as well(may be logarithmic in x axis?)
    # TODO: Make a function to calculate the estimated speed vs Recall.
    #self.__plot_roc3(file_name)

    self.__save_results(Path(self.config['results']['dir']))
    self.__save_all_params_file(Path(self.config['results']['dir']))
    self.results = []