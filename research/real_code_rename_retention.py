import argparse, json, os, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

from real_code_quotient_test import (
    DATA_BASE, CPP_PARSER, _text, download, load_jsonl, sample_by_label,
    encode, normalize_rows, project, random_basis
)


def rename_n_locals(code, seed, n_edits):
    if CPP_PARSER is None: return code
    src=code.encode('utf-8'); tree=CPP_PARSER.parse(src); root=tree.root_node
    rng=random.Random(9000+seed+97*n_edits)
    functions=[]
    def find_functions(node):
        if node.type=='function_definition': functions.append(node); return
        for c in node.children: find_functions(c)
    find_functions(root)
    replacements=[]; candidates=[]
    for fi,fn in enumerate(functions):
        declared=set()
        def collect(n):
            if n is not fn and n.type in ('function_definition','lambda_expression'): return
            if n.type=='identifier':
                p=n.parent.type if n.parent else ''
                gp=n.parent.parent.type if n.parent and n.parent.parent else ''
                if p in ('init_declarator','parameter_declaration','optional_parameter_declaration','variadic_parameter_declaration','catch_declaration','declaration'):
                    declared.add(_text(n,src))
                elif p in ('pointer_declarator','reference_declarator','array_declarator') and gp in ('declaration','parameter_declaration','init_declarator'):
                    declared.add(_text(n,src))
            for c in n.children: collect(c)
        collect(fn)
        declared=[x for x in sorted(declared) if x and not x.startswith('__') and x!='main']
        for name in declared: candidates.append((fi,fn,name))
    rng.shuffle(candidates); selected=candidates[:min(n_edits,len(candidates))]
    if not selected: return code
    selected_map={(fi,name):f'__irr_r_{seed}_{j}' for j,(fi,fn,name) in enumerate(selected)}
    for fi,fn in enumerate(functions):
        relevant={name:new for (ffi,name),new in selected_map.items() if ffi==fi}
        if not relevant: continue
        def replace(n):
            if n is not fn and n.type in ('function_definition','lambda_expression'): return
            if n.type=='identifier':
                name=_text(n,src); p=n.parent.type if n.parent else ''
                if name in relevant and p not in ('field_expression','qualified_identifier','namespace_identifier'):
                    replacements.append((n.start_byte,n.end_byte,relevant[name]))
            for c in n.children: replace(c)
        replace(fn)
    b=bytearray(src)
    for a,z,new in sorted(replacements,key=lambda x:x[0],reverse=True): b[a:z]=new.encode()
    out=b.decode('utf-8',errors='ignore')
    if CPP_PARSER.parse(out.encode()).root_node.has_error: return code
    return out


def top1_correct(q,c,labels):
    q=normalize_rows(q); c=normalize_rows(c); sim=q@c.T
    np.fill_diagonal(sim,-1e9)
    nn=np.argmax(sim,axis=1)
    lab=np.array(labels)
    return lab[nn]==lab


def retention(q,c,labels,base_correct):
    corr=top1_correct(q,c,labels)
    if base_correct.sum()==0: return np.nan, float(corr.mean())
    return float(corr[base_correct].mean()), float(corr.mean())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train=load_jsonl(download(DATA_BASE+'/train.jsonl',out/'train.jsonl'))
    test=load_jsonl(download(DATA_BASE+'/test.jsonl',out/'test.jsonl'))
    witnesses,_=sample_by_label(train,16,1,77)
    eval_rows,_=sample_by_label(test,12,20,88)
    wc=[r['code'] for r in witnesses]; ec=[r['code'] for r in eval_rows]; labels=[str(r['label']) for r in eval_rows]
    wren=[rename_n_locals(c,1000+i,8) for i,c in enumerate(wc)]
    banks={'base':ec}
    for n in [1,4,8]: banks[f'rename_{n}']=[rename_n_locals(c,2000+i,n) for i,c in enumerate(ec)]
    diag={k:float(np.mean([a!=b for a,b in zip(ec,v)])) for k,v in banks.items() if k!='base'}
    all_codes=wc+wren
    spans={'wbase':(0,len(wc)),'wren':(len(wc),2*len(wc))}; pos=2*len(wc)
    for k,v in banks.items(): spans[k]=(pos,pos+len(v)); all_codes+=v; pos+=len(v)
    emb,sec=encode(args.model,all_codes,batch=8,max_len=256)
    E={k:emb[a:z] for k,(a,z) in spans.items()}; base=E['base']
    rows=[]
    for K in [1,2,4,8,16]:
        D=E['wren'][:K]-E['wbase'][:K]
        u,s,vt=np.linalg.svd(D,full_matrices=False)
        cum=np.cumsum(s*s)/(np.sum(s*s)+1e-12); rank=int(np.searchsorted(cum,.9)+1)
        rank=max(1,min(rank,32,vt.shape[0])); U=vt[:rank].T; Ur=random_basis(base.shape[1],rank,500+K)
        X=E['wren'][:K]; Y=E['wbase'][:K]
        A=X.T@np.linalg.solve(X@X.T+1.0*np.eye(K),Y)
        methods={
            'raw':(base, lambda q:q),
            'quotient':(project(base,U), lambda q:project(q,U)),
            'random_rank':(project(base,Ur), lambda q:project(q,Ur)),
            'ridge_same_budget':(base, lambda q:normalize_rows(q@A)),
        }
        for method,(cand,f) in methods.items():
            bcorrect=top1_correct(f(base),cand,labels)
            for key in ['base','rename_1','rename_4','rename_8']:
                q=f(E[key]); ret,absacc=retention(q,cand,labels,bcorrect)
                rows.append({'model':args.model,'K':K,'rank':rank,'method':method,'transform':key,
                             'retention':ret,'absolute_top1_accuracy':absacc,'base_correct_n':int(bcorrect.sum()),
                             'embed_seconds':sec})
    df=pd.DataFrame(rows); df.to_csv(out/'rename_retention.csv',index=False)
    q=df[(df.K==4)&(df.transform=='rename_8')].set_index('method')
    raw=float(q.loc['raw','retention']); quo=float(q.loc['quotient','retention']); rnd=float(q.loc['random_rank','retention'])
    go=bool(quo>=raw+0.10 and quo>=rnd+0.08)
    verdict={'model':args.model,'go':go,'K':4,'rename8_raw_retention':raw,'rename8_quotient_retention':quo,
             'rename8_random_retention':rnd,'absolute_gain':quo-raw,'eval_n':len(ec),'diagnostics':diag,'embed_seconds':sec}
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('VERDICT',json.dumps(verdict,sort_keys=True),flush=True)
    print(df[df.K==4].pivot(index='transform',columns='method',values='retention').round(4).to_string(),flush=True)

if __name__=='__main__': main()
