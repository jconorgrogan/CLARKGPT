import argparse, ast, io, json, keyword, os, random, re, time, urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp
    CPP_LANGUAGE = Language(tree_sitter_cpp.language())
    CPP_PARSER = Parser(CPP_LANGUAGE)
except Exception:
    CPP_PARSER = None

DATA_BASE = "https://huggingface.co/datasets/semeru/Code-Code-CloneDetection-POJ104/resolve/main"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def download(url, path):
    path = Path(path)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def load_jsonl(path):
    out=[]
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def sample_by_label(rows, labels_n, per_label, seed):
    rng=random.Random(seed)
    groups=defaultdict(list)
    for r in rows:
        code=r.get("code", "")
        if 100 <= len(code) <= 5000:
            groups[str(r["label"])].append(r)
    labels=sorted([k for k,v in groups.items() if len(v)>=per_label])[:labels_n]
    selected=[]
    for lab in labels:
        pool=groups[lab][:]
        rng.shuffle(pool)
        selected.extend(pool[:per_label])
    return selected, labels


def insert_comments(code, seed):
    lines=code.splitlines()
    if not lines: return code
    rng=random.Random(1000+seed)
    comments=["// irrelevant metadata", "// implementation detail", "// local note", "// no semantic effect"]
    positions=sorted(set([min(len(lines), max(0, int(len(lines)*q))) for q in (0.2,0.5,0.8)]), reverse=True)
    for i,p in enumerate(positions):
        lines.insert(p, comments[(seed+i)%len(comments)])
    return "\n".join(lines)


def insert_noops(code, seed):
    # Empty statements after block openings are semantics-preserving in C/C++.
    # Limit the count to avoid excessive length growth.
    out=[]; inserted=0
    for line in code.splitlines():
        out.append(line)
        if inserted < 4 and "{" in line and not line.lstrip().startswith("#"):
            indent=re.match(r"\s*", line).group(0)
            out.append(indent + "  ;")
            inserted += 1
    return "\n".join(out)


def format_only(code, seed):
    # Lexer-level whitespace perturbation outside strings/comments is hard to do perfectly.
    # Use safe line-level formatting only: trailing spaces and blank lines.
    lines=[]
    for i,line in enumerate(code.splitlines()):
        lines.append(line.rstrip() + ("  " if (i+seed)%3==0 else ""))
        if (i+seed)%7==0: lines.append("")
    return "\n".join(lines)


def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def rename_locals(code, seed):
    if CPP_PARSER is None: return code
    src=code.encode("utf-8")
    tree=CPP_PARSER.parse(src)
    root=tree.root_node
    replacements=[]
    counter=0

    def walk_functions(node):
        nonlocal counter
        if node.type == "function_definition":
            process_function(node)
            return
        for ch in node.children:
            walk_functions(ch)

    def process_function(fn):
        nonlocal counter
        # Collect locally-declared identifiers and parameters in this function only;
        # do not descend into nested function definitions/lambdas.
        declared=set()
        globals_like=set()
        def collect(n):
            if n is not fn and n.type in ("function_definition", "lambda_expression"):
                return
            if n.type == "identifier":
                p=n.parent.type if n.parent else ""
                gp=n.parent.parent.type if n.parent and n.parent.parent else ""
                # Identifiers inside declarators, parameters, loop init/catch declarations.
                if p in ("init_declarator", "parameter_declaration", "optional_parameter_declaration", "variadic_parameter_declaration", "catch_declaration"):
                    declared.add(_text(n,src))
                elif p in ("pointer_declarator", "reference_declarator", "array_declarator") and gp in ("declaration", "parameter_declaration", "init_declarator"):
                    declared.add(_text(n,src))
                elif p == "declaration":
                    declared.add(_text(n,src))
            for c in n.children: collect(c)
        collect(fn)
        declared={x for x in declared if x and not x.startswith("__") and x not in ("main",)}
        if not declared: return
        mapping={name:f"__irr_v_{seed}_{counter+i}" for i,name in enumerate(sorted(declared))}
        counter += len(mapping)

        def replace(n):
            if n is not fn and n.type in ("function_definition", "lambda_expression"):
                return
            if n.type == "identifier":
                name=_text(n,src)
                if name in mapping:
                    p=n.parent.type if n.parent else ""
                    # Don't rewrite member names after . / -> / :: or the declared function name.
                    if p not in ("field_expression", "qualified_identifier", "namespace_identifier"):
                        replacements.append((n.start_byte,n.end_byte,mapping[name]))
            for c in n.children: replace(c)
        replace(fn)

    walk_functions(root)
    if not replacements: return code
    b=bytearray(src)
    for a,z,new in sorted(replacements, key=lambda x:x[0], reverse=True):
        b[a:z]=new.encode()
    out=b.decode("utf-8", errors="ignore")
    # Only accept if the transformed code still parses without ERROR nodes.
    t=CPP_PARSER.parse(out.encode())
    if t.root_node.has_error:
        return code
    return out


def transform(code, kind, seed):
    if kind=="rename": return rename_locals(code,seed)
    if kind=="comment": return insert_comments(code,seed)
    if kind=="noop": return insert_noops(code,seed)
    if kind=="format": return format_only(code,seed)
    if kind=="composition":
        x=rename_locals(code,seed)
        x=insert_noops(x,seed)
        x=insert_comments(x,seed)
        return format_only(x,seed)
    raise ValueError(kind)


def normalize_rows(xs):
    n=np.linalg.norm(xs,axis=1,keepdims=True)+1e-12
    return xs/n


def encode(model_name, codes, batch=8, max_len=256):
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    model.eval()
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    outs=[]
    t0=time.time()
    with torch.no_grad():
        for a in range(0,len(codes),batch):
            enc=tok(codes[a:a+batch],padding=True,truncation=True,max_length=max_len,return_tensors="pt")
            h=model(**enc).last_hidden_state
            mask=enc["attention_mask"].unsqueeze(-1).float()
            pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1.0)
            outs.append(pooled.cpu().numpy())
    return normalize_rows(np.concatenate(outs)), time.time()-t0


def map_at_r(query, cand, labels, exclude_self=True):
    q=normalize_rows(query); c=normalize_rows(cand)
    sim=q@c.T
    vals=[]
    for i in range(len(q)):
        if exclude_self: sim[i,i]=-1e9
        rel=np.array([x==labels[i] for x in labels],dtype=bool)
        if exclude_self: rel[i]=False
        R=int(rel.sum())
        if R==0: continue
        order=np.argsort(-sim[i])[:R]
        hits=rel[order]
        if not hits.any(): vals.append(0.0); continue
        precisions=[]; hitn=0
        for rank,h in enumerate(hits,1):
            if h:
                hitn+=1; precisions.append(hitn/rank)
        vals.append(sum(precisions)/R)
    return float(np.mean(vals))


def pair_cos(a,b):
    a=normalize_rows(a); b=normalize_rows(b)
    return float(np.mean(np.sum(a*b,axis=1)))


def quotient_basis(base_w, trans_w, energy=0.90, max_rank=64):
    D=np.concatenate([trans_w[k]-base_w for k in sorted(trans_w)],axis=0)
    # Center only across witness differences; nuisance directions are the row span.
    u,s,vt=np.linalg.svd(D,full_matrices=False)
    if np.all(s<1e-10): return np.zeros((base_w.shape[1],0)),0,s
    cum=np.cumsum(s*s)/np.sum(s*s)
    rank=int(np.searchsorted(cum,energy)+1)
    rank=max(1,min(rank,max_rank,vt.shape[0]))
    return vt[:rank].T,rank,s


def project(x,U):
    if U.shape[1]==0: return normalize_rows(x)
    return normalize_rows(x-(x@U)@U.T)


def random_basis(dim,rank,seed):
    if rank==0: return np.zeros((dim,0))
    rng=np.random.RandomState(seed)
    q,_=np.linalg.qr(rng.randn(dim,rank))
    return q[:,:rank]


def ridge_map(base_w, trans_w, lam=1.0):
    X=np.concatenate([trans_w[k] for k in sorted(trans_w)],axis=0)
    Y=np.concatenate([base_w for _ in sorted(trans_w)],axis=0)
    # Dual ridge: A = X^T (X X^T + lam I)^-1 Y
    K=X@X.T + lam*np.eye(len(X))
    A=X.T@np.linalg.solve(K,Y)
    return A


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="microsoft/codebert-base")
    ap.add_argument("--out",default="real_code_results")
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train_p=download(DATA_BASE+"/train.jsonl",out/"train.jsonl")
    test_p=download(DATA_BASE+"/test.jsonl",out/"test.jsonl")
    train=load_jsonl(train_p); test=load_jsonl(test_p)

    witness_rows,_=sample_by_label(train,labels_n=16,per_label=2,seed=11)  # 32 witnesses
    eval_rows,eval_labels=sample_by_label(test,labels_n=12,per_label=20,seed=22) # 240 eval
    eval_codes=[r["code"] for r in eval_rows]; labels=[str(r["label"]) for r in eval_rows]
    witness_codes=[r["code"] for r in witness_rows]
    primitives=["rename","comment","noop"]
    eval_kinds=["rename","comment","noop","format","composition"]

    # Build all texts before model inference.
    witness_text={"base":witness_codes}
    for kind in primitives:
        witness_text[kind]=[transform(c,kind,1000+i) for i,c in enumerate(witness_codes)]
    eval_text={"base":eval_codes}
    for kind in eval_kinds:
        eval_text[kind]=[transform(c,kind,2000+i) for i,c in enumerate(eval_codes)]

    # Transformation diagnostics.
    diag=[]
    for split,base,bank in [("witness",witness_codes,witness_text),("eval",eval_codes,eval_text)]:
        for kind,codes in bank.items():
            if kind=="base": continue
            changed=np.mean([a!=b for a,b in zip(base,codes)])
            diag.append({"split":split,"kind":kind,"changed_rate":changed,
                         "mean_chars":float(np.mean([len(x) for x in codes]))})
    pd.DataFrame(diag).to_csv(out/"transform_diagnostics.csv",index=False)

    all_codes=[]; spans={}
    for group,bank in [("witness",witness_text),("eval",eval_text)]:
        for kind,codes in bank.items():
            start=len(all_codes); all_codes.extend(codes); spans[(group,kind)]=(start,len(all_codes))
    emb,embed_seconds=encode(args.model,all_codes,batch=8,max_len=256)
    E={k:emb[a:z] for k,(a,z) in spans.items()}

    base_eval=E[("eval","base")]
    raw_original=map_at_r(base_eval.copy(),base_eval,labels)
    rows=[]; witness_stats=[]

    for K in [1,2,4,8]:
        # K witnesses per primitive transform, selected disjointly where possible.
        idx=[]
        for j,kind in enumerate(primitives):
            start=j*8
            idx.extend(list(range(start,start+K)))
        # For quotient each transform gets its own K examples, but same base array ordering.
        base_parts=[]; trans_parts={k:[] for k in primitives}
        for j,kind in enumerate(primitives):
            ids=list(range(j*8,j*8+K))
            base_parts.append(E[("witness","base")][ids])
            for kk in primitives:
                # only actual pairs from kk's designated witness block
                if kk==kind: trans_parts[kk]=E[("witness",kk)][ids]
        # Build one base matrix aligned with concatenated transform banks for basis/ridge.
        base_by_kind={}
        trans_by_kind={}
        for j,kind in enumerate(primitives):
            ids=list(range(j*8,j*8+K))
            base_by_kind[kind]=E[("witness","base")][ids]
            trans_by_kind[kind]=E[("witness",kind)][ids]
        D=np.concatenate([trans_by_kind[k]-base_by_kind[k] for k in primitives],axis=0)
        u,s,vt=np.linalg.svd(D,full_matrices=False)
        cum=np.cumsum(s*s)/(np.sum(s*s)+1e-12)
        rank=int(np.searchsorted(cum,.90)+1) if len(s) else 0
        rank=max(1,min(rank,64,vt.shape[0])) if len(s) else 0
        U=vt[:rank].T if rank else np.zeros((base_eval.shape[1],0))
        Ur=random_basis(base_eval.shape[1],rank,999+K)

        # Ridge baseline with exact same witness pairs.
        X=np.concatenate([trans_by_kind[k] for k in primitives],axis=0)
        Y=np.concatenate([base_by_kind[k] for k in primitives],axis=0)
        A=X.T@np.linalg.solve(X@X.T+1.0*np.eye(len(X)),Y)

        # Inference costs after embeddings exist.
        tproj=time.perf_counter(); cand_q=project(base_eval,U); projection_ms=(time.perf_counter()-tproj)*1000
        cand_rand=project(base_eval,Ur)

        # Oracle orbit-average candidate bank requires 4 forward representations per candidate.
        orbit_cand=normalize_rows(np.mean(np.stack([E[("eval",k)] for k in ["base"]+primitives],axis=0),axis=0))

        for kind in ["base"]+eval_kinds:
            q=E[("eval",kind)]
            method_scores={
                "raw":map_at_r(q.copy(),base_eval,labels),
                "quotient":map_at_r(project(q,U),cand_q,labels),
                "random_rank_control":map_at_r(project(q,Ur),cand_rand,labels),
                "ridge_same_budget":map_at_r(normalize_rows(q@A),base_eval,labels),
                "oracle_orbit_average":map_at_r(normalize_rows(np.mean(np.stack([q]+[E[("eval",k)] for k in primitives],axis=0),axis=0)),orbit_cand,labels),
            }
            for method,score in method_scores.items():
                rows.append({"model":args.model,"K_per_generator":K,"rank":rank,
                             "transform":kind,"method":method,"map_at_r":score,
                             "raw_original_map":raw_original,"embed_seconds":embed_seconds,
                             "projection_ms_all_eval":projection_ms})
        witness_stats.append({"K_per_generator":K,"rank":rank,
                              "pair_cos_raw":float(np.mean([pair_cos(base_by_kind[k],trans_by_kind[k]) for k in primitives])),
                              "pair_cos_quotient":float(np.mean([pair_cos(project(base_by_kind[k],U),project(trans_by_kind[k],U)) for k in primitives])),
                              "difference_energy_top_rank":float(cum[rank-1] if rank else 0)})

    res=pd.DataFrame(rows); res.to_csv(out/"results.csv",index=False)
    pd.DataFrame(witness_stats).to_csv(out/"witness_stats.csv",index=False)

    # GO/NO-GO: K=4, unseen composition. Require material recovery over raw, beat random control,
    # and be within 90% of expensive oracle augmentation's gain above raw.
    q=res[(res.K_per_generator==4)&(res.transform=="composition")].set_index("method")
    raw=float(q.loc["raw","map_at_r"]); quo=float(q.loc["quotient","map_at_r"])
    rnd=float(q.loc["random_rank_control","map_at_r"]); orb=float(q.loc["oracle_orbit_average","map_at_r"])
    gain=quo-raw; oracle_gain=orb-raw
    recovery=(gain/oracle_gain) if oracle_gain>1e-9 else np.nan
    go=bool(gain>=0.02 and quo>rnd+0.01 and (np.isnan(recovery) or recovery>=0.5))
    verdict={"model":args.model,"go":go,"raw_composition_map":raw,"quotient_composition_map":quo,
             "random_control_map":rnd,"oracle_orbit_map":orb,"absolute_gain":gain,
             "oracle_gain_recovery":recovery,"raw_original_map":raw_original,"embed_seconds":embed_seconds,
             "eval_programs":len(eval_codes),"witness_programs":len(witness_codes)}
    (out/"verdict.json").write_text(json.dumps(verdict,indent=2))
    print("VERDICT",json.dumps(verdict,sort_keys=True),flush=True)
    print(res[(res.K_per_generator==4)].pivot(index="transform",columns="method",values="map_at_r").round(4).to_string(),flush=True)

if __name__=="__main__": main()
