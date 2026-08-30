#!/usr/bin/env bash
set -u
OUT=${OUT:-alive_infer_results}
mkdir -p "$OUT/logs"
echo 'file,mode,exit_code,seconds,precondition_lines,valid_lines,output_bytes' > "$OUT/runs.csv"

mapfile -t FILES < <(find tests/infer/souper_tests -name '*.opt' | sort -V | head -n 24)
for mode in default no_features pre_features; do
  case "$mode" in
    default) FLAGS="--incompletes" ;;
    no_features) FLAGS="--incompletes --no-features" ;;
    pre_features) FLAGS="--incompletes --pre-features" ;;
  esac
  for file in "${FILES[@]}"; do
    base=$(basename "$file" .opt)
    log="$OUT/logs/${base}_${mode}.txt"
    start=$(date +%s)
    timeout 90s ./infer.py $FLAGS "$file" > "$log" 2>&1
    code=$?
    end=$(date +%s)
    secs=$((end-start))
    pres=$(grep -c '^Pre:' "$log" 2>/dev/null || true)
    valid=$(grep -Eic 'valid|weakest|precondition' "$log" 2>/dev/null || true)
    bytes=$(wc -c < "$log")
    echo "$file,$mode,$code,$secs,$pres,$valid,$bytes" >> "$OUT/runs.csv"
  done
done
python - <<'PY'
import csv, json, os
from collections import defaultdict
out=os.environ.get('OUT','alive_infer_results')
rows=list(csv.DictReader(open(out+'/runs.csv')))
summary=[]
for mode in sorted(set(r['mode'] for r in rows)):
    rr=[r for r in rows if r['mode']==mode]
    summary.append({
        'mode':mode,
        'files':len(rr),
        'completed_rate':sum(int(r['exit_code'])==0 for r in rr)/float(len(rr)),
        'timeout_rate':sum(int(r['exit_code'])==124 for r in rr)/float(len(rr)),
        'precondition_output_rate':sum(int(r['precondition_lines'])>0 for r in rr)/float(len(rr)),
        'median_seconds':sorted(int(r['seconds']) for r in rr)[len(rr)//2],
    })
with open(out+'/summary.json','w') as f: json.dump(summary,f,indent=2)
with open(out+'/summary.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
print(json.dumps(summary,indent=2))
PY
