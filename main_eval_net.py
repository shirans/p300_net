import torch
from preprocess.helpers import load_conf_eval, first_4_columns, first_4_col_names, get_device, build_model
from preprocess.preprocess import load_data
from run_fc.eval_model import evaluate

data_dir, model_dir = load_conf_eval()
train_loader, valid_loader = load_data(data_dir)

device = get_device()
model = build_model(device)
model.load_state_dict(torch.load(model_dir))
model.eval()

evaluate(train_loader, device, model)
evaluate(valid_loader, device, model)