from preprocess.helpers import load_conf_preprocess
from preprocess.preprocess import choose_columns_save_csv

data_dir, output_path = load_conf_preprocess()

# load the data
choose_columns_save_csv(data_dir, list(range(0, 7)), output_path)
