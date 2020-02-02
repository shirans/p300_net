from preprocess.helpers import load_conf_preprocess
from preprocess.preprocess import  load_data_choose_columns

# parameters
input_path, output_path = load_conf_preprocess()

# load the data, choose what columns to use, save output to csv
load_data_choose_columns(input_path, output_path)


