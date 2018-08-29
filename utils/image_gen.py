import numpy as np
import pandas as pd
import cv2

def random_transform(image, rotation_range=0, zoom_range=(1,1),
                     shift_range=0, random_flip=False):
    h,w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(zoom_range[0], zoom_range[1])
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w//2,h//2), rotation, scale)
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine(image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:,::-1]
    return result

class ImageGenerator(object):
    def __init__(self,x,y,
                 batch_size=32,
                 is_shuffle=True,
                 flip=True,
                 rebalance=False,
                 seed=0):
        np.random.seed(seed)
        self.images=x
        self.labels=y
        self.y=y
        self.rebalance=rebalance
        self.indices=None
        self.batch_size=batch_size
        self.shuffle=is_shuffle
        self.nb_samples=len(x)
        self.curr_pos=0
        self.threshold=0.5 if flip else 1
        self.steps_per_epoch=self.nb_samples//self.batch_size+1
        self._reset()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        end_pos=self.curr_pos+self.batch_size
        if end_pos<=self.nb_samples:
            indexes=self.indices[self.curr_pos:end_pos]
            self.curr_pos=end_pos
            if end_pos==self.nb_samples:
                self._reset()
        else:
            second_end_pos=self.batch_size-self.nb_samples+self.curr_pos
            indexes=self.indices[self.curr_pos:]+self.indices[:second_end_pos]
            self._reset()
        images=np.array([self._load_img(i) for i in self.images[indexes]])
        labels=np.array(self.labels[indexes])
        return images,labels
    
    def _load_img(self,path):
        img=cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=self._random_flip(img)
        img = img - 127.5
        img = img * 0.0078125
        return img
    
    def _random_flip(self,x):
        if np.random.random()>=self.threshold:
            return np.flip(x,1)
        return x
    
    def _reset(self):
        if self.rebalance:
            self._downsample()
        self.curr_pos=0
        indices=range(self.nb_samples)
        if self.shuffle:
            indices=np.random.permutation(indices)
        self.indices=list(indices)
        
    def _downsample(self):
        df=pd.DataFrame(self.y.copy())
        for col in df.columns:
            counts=df[col].value_counts()
            pos_counts=counts[1]
            neg_counts=counts[0]
            gap=pos_counts-neg_counts
            if gap!=0:
                target=1 if gap>0 else 0
                idx=np.random.choice(df[df[col]==target].index,size=abs(gap),replace=False)
                df.loc[idx,col]=-1
        self.labels=df.values
