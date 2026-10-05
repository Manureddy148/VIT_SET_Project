"""Requires owner/clinician-labeled relevance IDs. No synthetic benchmark claims."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.metrics import retrieval_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--judgments',required=True); p.add_argument('--output',default='reports/retrieval_metrics.json'); a=p.parse_args()
    queries=json.loads(Path(a.judgments).read_text())
    if not queries: raise ValueError('Empty judgments')
    report={}
    for strategy in ['shap','keyword']:
        rows=[retrieval_metrics(q[f'{strategy}_ranked_ids'], set(q['relevant_ids'])) for q in queries]
        report[strategy]={key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
    report['query_count']=len(queries)
    report['relative_ndcg_gain']=(report['shap']['ndcg']/report['keyword']['ndcg']-1) if report['keyword']['ndcg'] else None
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
