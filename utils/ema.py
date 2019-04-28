import torch
from torch import nn

class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow_params = {}
    
    def register(self, model, clear=False):
        if clear:
            self.shadow_params.clear()
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.data.clone()

    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                assert n in self.shadow_params
                new_param = self.decay*self.shadow_params[n]
                new_param += (1-self.decay)*p.data
                self.shadow_params[n] = new_param.clone()

if __name__=='__main__':
    class TestModel(nn.Module):
        def __init__(self, init_value):
            super(TestModel, self).__init__()
            self.linear1 = nn.Linear(6, 2, bias=False)
            nn.init.constant_(self.linear1.weight.data, init_value)
    model1 = TestModel(3)
    model2 = TestModel(init_value=5)
    for n,p in model1.named_parameters():
        print(n, p)
    ema = EMA(0.95)
    ema.register(model1, clear=True)
    print(ema.shadow_params)
    ema.update(model2)
    print(ema.shadow_params)
