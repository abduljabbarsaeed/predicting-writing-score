extract and add files from https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data
The required files are: train_logs.csv, train_scores.csv and test_logs.csv
Place all these files in the current working directory.
At an instance either training will run or testing.
If model file 'lstmModel.h5' is not there already then training should be run first to generate it by executing command: python run.py --trainortest 0
Then testing can be run by: python run.py --trainortest 1. This will generate test_scores.csv
