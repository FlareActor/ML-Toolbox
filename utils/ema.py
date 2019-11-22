import torch
import torch.nn as nn


class ExponentialMovingAverage(object):
    def __init__(self, decay=0.9995, dynamic=True):
        self.decay = decay
        self.dynamic = dynamic
        self.shadow_params = {}
        self.step = 0
    
    def register(self, model, clear=False):
        if clear:
            self.shadow_params.clear()
            self.step = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.data.clone()

    def update(self, model):
        self.step += 1
        if self.dynamic:
            decay = min((1 + self.step) / (10 + self.step), self.decay)
        else:
            decay = self.decay
            
        for n, p in model.named_parameters():
            if p.requires_grad:
                new_param = decay * self.shadow_params[n] + (1 - decay) * p.data
                self.shadow_params[n] = new_param
                
    def exchange(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                tmp = self.shadow_params[n]
                self.shadow_params[n] = p.data
                p.data = tmp
        return model

                
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
        
    ema = ExponentialMovingAverage(0.95)
    ema.register(model1, clear=True)
    print(ema.shadow_params)
    ema.update(model2)
    print(ema.shadow_params)
    
    print('\n')
    ema_model = ema.exchange(model1)
    for n,p in ema_model.named_parameters():
        print(n, p)
    
    model = ema.exchange(ema_model)
    for n,p in model.named_parameters():
        print(n, p)
