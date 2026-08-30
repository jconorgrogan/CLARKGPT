import argparse, itertools, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

ENVIRONMENTS=[0.1,0.2,0.9]
LABEL_NOISE=0.25
TRAIN_VAL_SPLIT=0.8
BATCH=128
LR=1e-3

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def xor(a,b): return (a-b).abs()
def bernoulli(p,n): return (torch.rand(n)<p).float()

def color_dataset(images, labels, env):
    y=(labels<5).float(); y=xor(y,bernoulli(LABEL_NOISE,len(y)))
    x=torch.stack([images,images],dim=1)
    colors=xor(y,bernoulli(env,len(y)))
    x[torch.arange(len(x)),(1-colors).long()]*=0
    return x.float()/255.0,y.long()

def build(seed,root):
    seed_all(seed)
    tr=MNIST(root,train=True,download=True); te=MNIST(root,train=False,download=True)
    imgs=torch.cat([tr.data,te.data]); labels=torch.cat([tr.targets,te.targets])
    p=torch.randperm(len(imgs)); imgs=imgs[p]; labels=labels[p]
    envs=[]; raws=[]
    for i,e in enumerate(ENVIRONMENTS):
        rx,ry=imgs[i::3],labels[i::3]
        envs.append(color_dataset(rx,ry,e)); raws.append((rx,ry))
    train_parts=[]; raw_train=[]
    for i in range(2):
        x,y=envs[i]; rx,ry=raws[i]
        nt=math.ceil(len(x)*TRAIN_VAL_SPLIT); nv=len(x)-nt
        q=torch.randperm(len(x),generator=torch.Generator().manual_seed(42))
        ti=q[:nt]
        train_parts.append((x[ti],y[ti])); raw_train.append((rx[ti],ry[ti]))
    return (torch.cat([a for a,_ in train_parts]),torch.cat([b for _,b in train_parts]),
            envs[2][0],envs[2][1],torch.cat([a for a,_ in raw_train]).float()/255,
            torch.cat([b for _,b in raw_train]))

def infer_one_pair(raw,raw_labels,seed):
    idx=torch.where((raw_labels<5)==1)[0]
    i=idx[torch.randperm(len(idx),generator=torch.Generator().manual_seed(10000+seed))[0]]
    im=raw[i:i+1]; z=torch.zeros_like(im)
    left=torch.stack([im,z],dim=1); right=torch.stack([z,im],dim=1)
    scores=[]
    for perm in itertools.permutations(range(2)):
        p=torch.tensor(perm)
        r=float(((left[:,p]-right)**2).mean()+((right[:,p]-left)**2).mean())
        scores.append((r,perm))
    scores.sort()
    return scores[0][1],scores

def project(x,perm):
    p=torch.tensor(perm)
    return x+x[:,p]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(2,64,3,1,1); self.c2=nn.Conv2d(64,128,3,2,1)
        self.c3=nn.Conv2d(128,128,3,1,1); self.c4=nn.Conv2d(128,128,3,1,1)
        self.n0=nn.GroupNorm(8,64); self.n1=nn.GroupNorm(8,128)
        self.n2=nn.GroupNorm(8,128); self.n3=nn.GroupNorm(8,128)
        self.h=nn.Linear(128,2)
    def forward(self,x):
        x=self.n0(F.relu(self.c1(x))); x=self.n1(F.relu(self.c2(x)))
        x=self.n2(F.relu(self.c3(x))); x=self.n3(F.relu(self.c4(x)))
        return self.h(F.adaptive_avg_pool2d(x,1).reshape(len(x),-1))

@torch.no_grad()
def acc(m,x,y):
    m.eval(); c=0
    for i in range(0,len(x),1024): c+=int((m(x[i:i+1024]).argmax(1)==y[i:i+1024]).sum())
    return c/len(y)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--epochs',type=int,default=18); ap.add_argument('--out',default='results')
    a=ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    t=time.time(); xtr,ytr,xt,yt,raw,rawy=build(a.seed,'mnist_data')
    perm,scores=infer_one_pair(raw,rawy,a.seed)
    xtr=project(xtr,perm); xt=project(xt,perm)
    seed_all(a.seed); m=CNN(); opt=torch.optim.Adam(m.parameters(),lr=LR)
    g=torch.Generator().manual_seed(20000+a.seed)
    for _ in range(a.epochs):
        m.train(); order=torch.randperm(len(xtr),generator=g)
        for i in range(0,len(order),BATCH):
            q=order[i:i+BATCH]; opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(m(xtr[q]),ytr[q]); loss.backward(); opt.step()
    test=acc(m,xt,yt)
    result={'seed':a.seed,'epochs':a.epochs,'pair_budget':1,'learned_perm':list(perm),'candidate_scores':[{'residual':r,'perm':list(p)} for r,p in scores],'test_acc':test,'seconds':time.time()-t}
    Path(a.out,f'seed_{a.seed}.json').write_text(json.dumps(result,indent=2)); print('RESULT',json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__': main()
