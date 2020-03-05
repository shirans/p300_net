import torch
from torchsummary import summary
from torchvision import models

from preprocess.helpers import load_conf_eval, get_device, get_model
from preprocess.preprocess import build_dataloader
from run_fc.eval_model import evaluate
from run_fc.metadata import Metadata

data_dir, model_dir, network = load_conf_eval()
metadata = Metadata(None)
train_loader, valid_loader = build_dataloader(data_dir, 64,metadata, False)

device = get_device()
model = get_model(device, network)
model.load_state_dict(torch.load(model_dir))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

# summary(vgg, (3, 224, 224))

metadata = Metadata(None)
print("working with model:{}".format(model_dir))
evaluate(train_loader, device, model, 'train', metadata)
evaluate(valid_loader, device, model, 'validation', metadata)
summary(model.type(torch.FloatTensor), (1, 6, 217))
print("Done!")
