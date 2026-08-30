import argparse, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from real_code_quotient_test import DATA_BASE, download, load_jsonl, encode, normalize_rows, project, random_basis
from real_code_rename_retention import rename_n_locals

def pick(rows, labels_n, per_label, seed):
    rng=random.Random(seed); groups=defaultdict(list)
    for r in rows:
        c=r.get('code','')
        if 100<=len(c)<=3500: groups[str(r['label'])].append(r)
    labels=sorted([k for k,v in groups.items() if len(v)>=per_label])[:labels_n]
    out=[]
    for lab in labels:
        p=groups[lab][:]; rng.shuffle(p); out.extend(p[:per_label])
    return out

def norm(x): return normalize_rows(x)

def top1(q,c,labels):
    s=norm(q)@norm(c).T; np.fill_diagonal(s,-1e9); nn=s.argmax(1); lab=np.array(labels); return lab[nn]==lab

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train=load_jsonl(download(DATA_BASE+'/train.jsonl',out/'train.jsonl')); test=load_jsonl(download(DATA_BASE+'/test.jsonl',out/'test.jsonl'))
    # 8 unrelated train-class witnesses; use the first four that actually change.
    wr=pick(train,16,2,991); wc=[r['code'] for r in wr]
    w8=[rename_n_locals(c,10000+i,8) for i,c in enumerate(wc)]
    changed=[i for i,(a,b) in enumerate(zip(wc,w8)) if a!=b]
    if len(changed)<4: raise RuntimeError(f'Only {len(changed)} changed witness programs')
    wi=changed[:4]; wb=[wc[i] for i in wi]; wt=[w8[i] for i in wi]

    er=pick(test,10,12,992); ec=[r['code'] for r in er]; labels=[str(r['label']) for r in er]
    banks={'base':ec}
    for n in (1,4,8): banks[f'r{n}']=[rename_n_locals(c,20000+i,n) for i,c in enumerate(ec)]
    # Keep only examples where 8-renaming actually changed code, but preserve >=8 examples/label when possible.
    keep=[i for i,(a,b) in enumerate(zip(ec,banks['r8'])) if a!=b]
    if len(keep)<50: raise RuntimeError(f'Only {len(keep)} eval programs changed under rename8')
    ec=[ec[i] for i in keep]; labels=[labels[i] for i in keep]
    for k in list(banks): banks[k]=[banks[k][i] for i in keep]

    texts=wb+wt; spans={'wb':(0,len(wb)),'wt':(len(wb),2*len(wb))}; pos=2*len(wb)
    for k,v in banks.items(): spans[k]=(pos,pos+len(v)); texts+=v; pos+=len(v)
    Eall,secs=encode(args.model,texts,batch=16,max_len=160); E={k:Eall[a:z] for k,(a,z) in spans.items()}
    D=E['wt']-E['wb']; u,s,vt=np.linalg.svd(D,full_matrices=False); cum=np.cumsum(s*s)/(np.sum(s*s)+1e-12); rank=int(np.searchsorted(cum,.90)+1); rank=max(1,min(rank,4))
    U=vt[:rank].T; Ur=random_basis(Eall.shape[1],rank,4242)
    X,Y=E['wt'],E['wb']; A=X.T@np.linalg.solve(X@X.T+1.0*np.eye(len(X)),Y)
    base=E['base']
    methods={'raw':(base,lambda q:q),'quotient':(project(base,U),lambda q:project(q,U)),'random_rank':(project(base,Ur),lambda q:project(q,Ur)),'ridge_same_budget':(base,lambda q:norm(q@A))}
    rows=[]
    for m,(cand,f) in methods.items():
        bc=top1(f(base),cand,labels)
        for k in ['base','r1','r4','r8']:
            corr=top1(f(E[k]),cand,labels); ret=float(corr[bc].mean()) if bc.any() else float('nan')
            rows.append({'model':args.model,'method':m,'transform':k,'retention':ret,'absolute_top1':float(corr.mean()),'base_correct_n':int(bc.sum()),'rank':rank,'eval_n':len(labels)})
    df=pd.DataFrame(rows); df.to_csv(out/'rapid_results.csv',index=False)
    q=df[df.transform=='r8'].set_index('method'); raw=float(q.loc['raw','retention']); quo=float(q.loc['quotient','retention']); rnd=float(q.loc['random_rank','retention']); ridge=float(q.loc['ridge_same_budget','retention'])
    # strict but reachable: +5 points, beat random by 3, and no >3 point loss in base absolute top1.
    b=df[df.transform=='base'].set_index('method'); base_loss=float(b.loc['raw','absolute_top1']-b.loc['quotient','absolute_top1'])
    go=bool(quo>=raw+.05 and quo>=rnd+.03 and base_loss<=.03)
    verdict={'model':args.model,'go':go,'raw_r8_retention':raw,'quotient_r8_retention':quo,'random_r8_retention':rnd,'ridge_r8_retention':ridge,'gain':quo-raw,'base_top1_loss':base_loss,'rank':rank,'witness_pairs':4,'eval_n':len(labels),'embedding_seconds':secs}
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2)); print('VERDICT',json.dumps(verdict,sort_keys=True)); print(df.pivot(index='transform',columns='method',values='retention').round(4).to_string())
if __name__=='__main__': main()
