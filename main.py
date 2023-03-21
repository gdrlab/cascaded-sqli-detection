import configparser
from templates import DataManager, logger
from experiments import TestManager


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
