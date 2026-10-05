"""Real abstracts only. HTTPS verification, batched requests, no synthetic fallback."""
import argparse,json,time,os,urllib.request,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path
BASE='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
QUERIES={'diabetes':'diabetes prognosis risk factors', 'heart_disease':'coronary heart disease prognosis risk factors','pneumonia':'pneumonia prognosis risk factors'}
def get(endpoint,params):
    params = {**params, 'tool': 'VITMedicalAIResearch'}
    if os.getenv('NCBI_CONTACT_EMAIL'): params['email'] = os.environ['NCBI_CONTACT_EMAIL']
    time.sleep(.4)  # below NCBI's unauthenticated 3 requests/sec limit
    req=urllib.request.Request(BASE+endpoint+'?'+urllib.parse.urlencode(params),headers={'User-Agent':'VITResearch/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=60) as r:return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(2**(attempt+1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=500);p.add_argument('--output',default='data/processed/pubmed_verified.json');a=p.parse_args()
    docs=[]; counts={}
    for domain,term in QUERIES.items():
        ids=json.loads(get('esearch.fcgi',{'db':'pubmed','retmode':'json','retmax':a.limit*2,'term':term}))['esearchresult']['idlist']
        domain_docs=[]
        for offset in range(0,len(ids),100):
            root=ET.fromstring(get('efetch.fcgi',{'db':'pubmed','retmode':'xml','id':','.join(ids[offset:offset+100])}))
            for article in root.findall('.//PubmedArticle'):
                abstract=' '.join(''.join(t.itertext()) for t in article.findall('.//Abstract/AbstractText'))
                pmid=article.findtext('.//MedlineCitation/PMID')
                if not abstract or not pmid:continue
                title=''.join(article.find('.//ArticleTitle').itertext()) if article.find('.//ArticleTitle') is not None else ''
                domain_docs.append({'id':f'{domain}_{pmid}','text':title+'. '+abstract,'metadata':{'domain':domain,'pmid':pmid,'title':title,'source':f'PubMed:{pmid}','url':f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/','synthetic':False}})
                if len(domain_docs)>=a.limit:break
            if len(domain_docs)>=a.limit:break
        counts[domain]=len(domain_docs);docs.extend(domain_docs)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(docs,indent=2))
    print(json.dumps({'counts':counts,'total':len(docs),'target_per_domain':a.limit}))
    if any(n<a.limit for n in counts.values()):raise SystemExit('Insufficient real abstracts; see saved counts')
if __name__=='__main__':main()
