"""Actual RAGAS evaluator over owner-reviewed reports, no auto-generated ground truth.
JSON list: user_input,response,retrieved_contexts. Requires separately configured evaluator.
"""
import argparse,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);p.add_argument('--output',default='reports/ragas_metrics.json');a=p.parse_args()
    rows=json.loads(Path(a.reports).read_text())
    if not rows or any(not r.get('retrieved_contexts') or not r.get('response') for r in rows):raise ValueError('Reports and source contexts required')
    from ragas import EvaluationDataset,evaluate
    from ragas.metrics import Faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq
    if not os.getenv('GROQ_API_KEY'):raise RuntimeError('Set GROQ_API_KEY securely in local environment; no score produced')
    evaluator=LangchainLLMWrapper(ChatGroq(model=os.getenv('RAGAS_GROQ_MODEL','llama-3.3-70b-versatile'),temperature=0))
    result=evaluate(EvaluationDataset.from_list(rows),metrics=[Faithfulness(llm=evaluator)])
    frame=result.to_pandas();out={'report_count':len(frame),'faithfulness_mean':float(frame.faithfulness.mean()),'individual':frame.faithfulness.tolist()}
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
