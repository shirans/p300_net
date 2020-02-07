from preprocess.helpers import load_conf_preprocess
from preprocess.preprocess import choose_columns_save_csv

data_dir, output_path = load_conf_preprocess()

choose_columns_save_csv(data_dir, output_path, [42, 24, 29, 44, 45])
print("Done!")
