import torch
import os
def load_model(model_path,device):
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()
    print(model_path + ' loaded')
    return model

def load_models(models_dir,device):
    global pnet,rnet,onet,softmax_p,softmax_r,softmax_o
    pnet = load_model(os.path.join(models_dir, 'PNet.pth'),device)
    rnet = load_model(os.path.join(models_dir, 'RNet.pth'),device)
    onet = load_model(os.path.join(models_dir, 'ONet.pth'),device)