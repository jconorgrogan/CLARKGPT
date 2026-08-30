import json, math, os, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path(os.environ.get("OUT", "longmemeval_proxy_results"))
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "data"

STOP = set("a an and are as at be by for from has have how i in is it of on or our should that the there this to was what when where which who will with you your answer answers correct explicitly final wrapped boxed please concise must say says".split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_line(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    # Collapse highly volatile ids/timestamps while preserving textual distinctions.
    s = re.sub(r"\b\d{5,}\b", "<NUM>", s)
    s = re.sub(r"[0-9a-f]{12,}", "<ID>", s, flags=re.I)
    return s


def state_text(state):
    parts = []
    for key in ["url", "action", "thought"]:
        value = state.get(key)
        if value:
            parts.append(f"{key.upper()}: {norm_line(value)}")
    tree = state.get("accessibility_tree") or ""
    if tree:
        parts.extend(norm_line(x) for x in str(tree).splitlines() if norm_line(x))
    return parts


def full_memory(tr):
    parts = [f"GOAL: {norm_line(tr.get('goal',''))}", f"OUTCOME: {tr.get('outcome','')}"]
    for state in tr.get("states", []):
        parts.append(f"STATE {state.get('state_index','?')}")
        parts.extend(state_text(state))
    return "\n".join(parts)


def distinction_memory(tr):
    # Retain changes: action/thought, URL transitions, and lines entering/leaving the observed state.
    parts = [f"GOAL: {norm_line(tr.get('goal',''))}", f"OUTCOME: {tr.get('outcome','')}"]
    previous = set()
    seen_events = set()
    prev_url = None
    states = tr.get("states", [])
    for idx, state in enumerate(states):
        lines = state_text(state)
        current = set(lines)
        url = norm_line(state.get("url") or "")
        events = []
        if idx == 0:
            events.extend(lines)
        else:
            if url and url != prev_url:
                events.append(f"URL CHANGED: {prev_url} -> {url}")
            action = state.get("action")
            thought = state.get("thought")
            if action:
                events.append(f"ACTION: {norm_line(action)}")
            if thought:
                events.append(f"THOUGHT: {norm_line(thought)}")
            added = sorted(current - previous)
            removed = sorted(previous - current)
            events.extend(f"APPEARED: {x}" for x in added)
            # Removed landmarks matter for dynamic-state and gotcha questions.
            events.extend(f"DISAPPEARED: {x}" for x in removed[:80])
        unique = []
        for e in events:
            if e and e not in seen_events:
                seen_events.add(e); unique.append(e)
        if unique:
            parts.append(f"CHANGE {state.get('state_index',idx)}")
            parts.extend(unique)
        previous = current
        prev_url = url
    return "\n".join(parts)


def unique_line_memory(tr):
    seen = set(); parts = [f"GOAL: {norm_line(tr.get('goal',''))}", f"OUTCOME: {tr.get('outcome','')}"]
    for state in tr.get("states", []):
        for line in state_text(state):
            if line not in seen:
                seen.add(line); parts.append(line)
    return "\n".join(parts)


def uniform_sample(text, budget):
    if len(text) <= budget: return text
    if budget <= 0: return ""
    chunks = 32
    take = max(1, budget // chunks)
    starts = np.linspace(0, max(0, len(text)-take), chunks).astype(int)
    out = "\n".join(text[s:s+take] for s in starts)
    return out[:budget]


def headtail(text, budget):
    if len(text) <= budget: return text
    a = budget // 2
    return text[:a] + "\n" + text[-(budget-a):]


def chunks(text, size=5000, overlap=300):
    if not text: return [""]
    out=[]; start=0
    while start < len(text):
        out.append(text[start:start+size])
        if start+size >= len(text): break
        start += size-overlap
    return out


def answer_tokens(answer):
    toks = re.findall(r"[a-z0-9]+", str(answer).lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


def normalized_phrase(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def make_index(docs):
    texts=[]; tids=[]; by_tid=defaultdict(list)
    for tid,text in docs.items():
        for c in chunks(text):
            by_tid[tid].append(len(texts)); tids.append(tid); texts.append(c)
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1,2), min_df=1,
        max_features=80000, sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, texts, tids, by_tid


def retrieve(index, query, allowed_ids, char_budget):
    vectorizer, matrix, texts, tids, by_tid = index
    cand=[]
    for tid in allowed_ids: cand.extend(by_tid.get(tid, []))
    if not cand: return "", 0.0
    t0=time.perf_counter()
    q=vectorizer.transform([query])
    scores=(matrix[cand] @ q.T).toarray().ravel()
    order=np.argsort(-scores)
    selected=[]; used=0
    for j in order:
        text=texts[cand[j]]
        remaining=char_budget-used
        if remaining<=0: break
        selected.append(text[:remaining]); used += min(len(text),remaining)
    return "\n".join(selected), time.perf_counter()-t0


def main():
    t0=time.time()
    snapshot_download(
        repo_id="xiaowu0162/longmemeval-v2", repo_type="dataset",
        local_dir=str(DATA),
        allow_patterns=["questions.jsonl","trajectories.jsonl","haystacks/lme_v2_small.json"],
    )
    questions=list(read_jsonl(DATA/"questions.jsonl"))
    hay=json.loads((DATA/"haystacks/lme_v2_small.json").read_text())
    wanted=set(x for ids in hay.values() for x in ids)
    trajectories={}
    for row in read_jsonl(DATA/"trajectories.jsonl"):
        if row.get("id") in wanted:
            trajectories[row["id"]]=row
    if wanted-set(trajectories):
        raise RuntimeError(f"Missing trajectories: {len(wanted-set(trajectories))}")

    raw={tid:full_memory(tr) for tid,tr in trajectories.items()}
    distinction={tid:distinction_memory(tr) for tid,tr in trajectories.items()}
    unique={tid:unique_line_memory(tr) for tid,tr in trajectories.items()}
    uniform={tid:uniform_sample(raw[tid],len(distinction[tid])) for tid in raw}
    ht={tid:headtail(raw[tid],len(distinction[tid])) for tid in raw}
    methods={"raw":raw,"distinction_delta":distinction,"unique_lines":unique,
             "uniform_matched":uniform,"headtail_matched":ht}

    storage=[]
    raw_chars=sum(map(len,raw.values()))
    for name,docs in methods.items():
        chars=sum(map(len,docs.values()))
        storage.append({"method":name,"stored_chars":chars,"ratio_to_raw":chars/raw_chars,
                        "trajectories":len(docs)})
    storage_df=pd.DataFrame(storage)
    storage_df.to_csv(OUT/"storage.csv",index=False)

    indexes={name:make_index(docs) for name,docs in methods.items()}

    # Text-only, deterministic-evaluator questions whose answers have lexical support somewhere
    # in the official small haystack. This measures evidence preservation/retrieval, not QA reasoning.
    eligible=[]
    raw_token_sets={tid:set(answer_tokens(text)) for tid,text in raw.items()}
    for q in questions:
        if q.get("image") is not None: continue
        if str(q.get("eval_function","")).startswith("llm_"): continue
        at=answer_tokens(q.get("answer",""))
        if not at or len(str(q.get("answer","")))>300: continue
        union=set()
        for tid in hay[q["id"]]: union |= raw_token_sets.get(tid,set())
        oracle_recall=len(set(at)&union)/len(set(at))
        if oracle_recall < 0.60: continue
        eligible.append((q,at,oracle_recall))

    rows=[]
    budgets=[4000,8000,16000,32000]
    for q,at,oracle_recall in eligible:
        phrase=normalized_phrase(q.get("answer",""))
        for method,index in indexes.items():
            for budget in budgets:
                context,latency=retrieve(index,q["question"],hay[q["id"]],budget)
                ct=set(answer_tokens(context))
                recall=len(set(at)&ct)/len(set(at))
                phrase_hit=int(bool(phrase) and phrase in normalized_phrase(context))
                rows.append({
                    "question_id":q["id"],"domain":q["domain"],"question_type":q["question_type"],
                    "method":method,"context_char_budget":budget,
                    "answer_token_recall":recall,"exact_answer_phrase_hit":phrase_hit,
                    "oracle_lexical_recall":oracle_recall,"query_latency_seconds":latency,
                })
    runs=pd.DataFrame(rows)
    runs.to_csv(OUT/"runs.csv",index=False)
    summary=(runs.groupby(["method","context_char_budget"])
             .agg(questions=("question_id","nunique"),
                  mean_answer_token_recall=("answer_token_recall","mean"),
                  exact_answer_phrase_hit_rate=("exact_answer_phrase_hit","mean"),
                  median_query_latency_ms=("query_latency_seconds",lambda x:1000*float(np.median(x))))
             .reset_index()
             .merge(storage_df[["method","ratio_to_raw","stored_chars"]],on="method",how="left"))
    summary.to_csv(OUT/"summary.csv",index=False)

    # Matched-budget headline against uniform/head-tail controls.
    headline=summary[summary.context_char_budget==16000].sort_values("mean_answer_token_recall",ascending=False)
    metadata={
        "runtime_seconds":time.time()-t0,
        "official_questions":len(questions),"official_small_haystack_trajectories":len(trajectories),
        "eligible_lexically_supported_text_questions":len(eligible),
        "raw_stored_chars":raw_chars,
        "note":"Official LME-V2 text and small haystacks; retrieval/evidence-preservation proxy only, not official fixed-reader accuracy or leaderboard LAFS. Memory compression is built before questions."
    }
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print("HEADLINE_16K")
    print(headline.to_string(index=False))
    print("STORAGE")
    print(storage_df.to_string(index=False))
    print("META",json.dumps(metadata,sort_keys=True))


if __name__=="__main__":
    main()
