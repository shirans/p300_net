import torch

from preprocess.helpers import load_conf_eval, get_device, build_model
from preprocess.preprocess import data_to_raw
from run_fc.eval_model import evaluate

data_dir, model_dir = load_conf_eval()
train_loader, valid_loader = data_to_raw(data_dir)

device = get_device()
model = build_model(device)
model.load_state_dict(torch.load(model_dir))
model.eval()

print("working with model:{}".format(model_dir))
evaluate(train_loader, device, model, 'train')
evaluate(valid_loader, device, model,'validation')
print("Done!")
