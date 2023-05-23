from templates import FeatureExtractor
from classical_models import Classical_Model
from pprint import pprint
from experiments import evaluate

threshold = 0.05 #prediction
scale_pos_weight = 5000.0

feature_method = 'tf-idf_ngram'
model_name = 'XGBoost'

# Train and evaluate the first stage
def first_stage():
  feature_extractor = FeatureExtractor(feature_method)
  start_time = time.time()
  feature_extractor.extract_features(
      data_manager.x_train, data_manager.x_test)
  stop_time = time.time()
  extraction_time = ((stop_time - start_time)*1000 
                    / (len(data_manager.x_train) + len(data_manager.x_test)) )

  model = Classical_Model(model_name, scale_pos_weight=scale_pos_weight)
  model.feature_method = feature_extractor.method
  start_time = time.time()
  model.fit(
      feature_extractor.features['train'], 
      data_manager.y_train) #scale_pos_weight=5.0
  stop_time = time.time()
  training_time = (stop_time - start_time)*1000 / feature_extractor.features['train'].shape[0]


  start_time = time.time()
  first_stage_y_pred = model.predict(feature_extractor.features['test'], threshold=threshold)
  stop_time = time.time()

  testing_time = (stop_time - start_time)*1000 / feature_extractor.features['test'].shape[0]

  notes = {'feature_method': feature_extractor.method,'model': model_name, 
            'seed': data_manager.notes['seed'], 
            'split_ratio': data_manager.notes['split_ratio'],
            'train_size': data_manager.notes['train_size'],
            'test_size': data_manager.notes['test_size'],
            'extraction_time':extraction_time, 'feature_size': feature_extractor.features['train'].shape[1],
            'train_time': training_time,
            'pred_time': testing_time, 'dataset': config['dataset']['file'],
          'scale_pos_weight':scale_pos_weight,
          'threshold': threshold
          }
  # Save results to csv file

  result = evaluate(data_manager.y_test, first_stage_y_pred, notes=notes)
  pprint(result)
  #save_results([result], proposed_test_results_file)

  # Extract the positive predicitions for the second stage
  first_stage_positive_preds = data_manager.x_test[(first_stage_y_pred == 1)]
  first_stage_positive_preds_true_labels = data_manager.y_test[(first_stage_y_pred == 1)]
  fs_pos_len = len(first_stage_positive_preds)
  nof_test_samples = len(data_manager.x_test)
  fs_rat = fs_pos_len/nof_test_samples
  print(f"Positive predicitions in the first stage: {fs_pos_len} out of {nof_test_samples}. Ratio:{fs_rat}")

  return first_stage_positive_preds, first_stage_positive_preds_true_labels


first_stage_y_pred, first_stage_positive_preds_true_labels = first_stage()
