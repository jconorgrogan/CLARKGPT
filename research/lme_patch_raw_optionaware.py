from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    class_start = text.index("class DistinctionGraphMemory")
    query_start = text.index("    def query(", class_start)
    raw_start = text.index('        if self.mode == "raw":\n', query_start)
    state_start = text.index("        if self.state_matrix is None", raw_start)

    replacement = (
        '        if self.mode == "raw":\n'
        '            if self.raw_matrix is None or not self.raw_texts:\n'
        '                return []\n'
        '            stem, options = self._query_variants(query)\n'
        '            variants = [stem] + options if options else [query]\n'
        '            query_matrix = self.vectorizer.transform(variants)\n'
        '            scores_all = (self.raw_matrix @ query_matrix.T).toarray()\n'
        '            scores = self._priority(scores_all)\n'
        '            order = np.argsort(-scores, kind="stable")[: self.top_k]\n'
        '            return [\n'
        '                {"type": "text", "value": self.raw_texts[int(index)]}\n'
        '                for index in order\n'
        '                if self.raw_texts[int(index)].strip()\n'
        '            ]\n'
    )
    patched = text[:raw_start] + replacement + text[state_start:]
    if patched == text:
        raise RuntimeError("Raw retrieval patch produced no change")
    path.write_text(patched, encoding="utf-8")
    print(f"Patched option-aware raw retrieval: {path}")


if __name__ == "__main__":
    main()
