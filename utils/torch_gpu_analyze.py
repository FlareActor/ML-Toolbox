import torch
import torch.nn as nn


class GPUAnalyzeModule(nn.Module):
    def __init__(self, name, module):
        super(GPUAnalyzeModule, self).__init__()
        self.name = name
        self.module = module
        
    def forward(self, x):
        m0 = torch.cuda.memory_allocated()
        outputs = self.module(x)
        m1 = torch.cuda.memory_allocated()
        gpu_usage = (m1-m0) / 1024 / 1024
        global total_gpu_usage
        total_gpu_usage += gpu_usage
        print('{}:\t{:.2f}MB'.format(self.name.rjust(30), gpu_usage))
        return outputs
    
    
def make_gpu_analysable(model):
    for n, m in vars(model)['_modules'].items():
        setattr(model, n, GPUAnalyzeModule(n, m))
        

if __name__=='__main__':
    import numpy as np
    import sys
    import pdb
    sys.path.append('/users/wangdexun/SpeakerRecognition/speaker_recognition/model_zoo')
    from torch_res34 import KitModel
    
    model = KitModel()
    device = torch.device('cuda')
    model.to(device)
    make_gpu_analysable(model)
    
    x = torch.from_numpy(np.random.rand(64,1,257,250)).type(torch.float32).to(device)
    total_gpu_usage = 0
    torch.cuda.empty_cache()
    model(x)
    print('{}:\t{:.2f}MB'.format('Total'.rjust(30), total_gpu_usage))
    
    pdb.set_trace()
    
    
    
  