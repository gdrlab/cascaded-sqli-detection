import configparser
from templates import DataManager, logger
from experiments import TestManager
import sys, getopt


def main(argv):
  opts, args = getopt.getopt(argv,"ho:",["ofile="])
  outputfile = ''
  for opt, arg in opts:
    if opt == '-h':
      print ('main.py -o <output_file_name>')
      sys.exit()
    elif opt in ("-o", "--ofile"):
      outputfile = arg

  config = configparser.ConfigParser()
  config.read('config.ini')
  data_manager = DataManager(config)
  test_manager = TestManager(data_manager=data_manager,config=config, output_file_name=outputfile)

  feature_methods = [method.strip() for method in config.get('feature_methods', 'methods').split(',')]
  classic_models = [model.strip() for model in config.get('models', 'classic_models').split(',')]
  ensemble_models = [model.strip() for model in config.get('models', 'ensemble_models').split(',')]

  seed_idx = 0
  while (data_manager.split_data(seed_idx=seed_idx)): #while there are more seeds
    print(f'Running the tests for seed: {data_manager.seed}')
    test_manager.run_tests(feature_methods, classic_models, ensemble_models)
    #test_manager.run_first_stage_tests() #DEBUG remove that
    seed_idx += 1

  return test_manager.output_file_name
  


if __name__ == "__main__":
  main(sys.argv[1:])
