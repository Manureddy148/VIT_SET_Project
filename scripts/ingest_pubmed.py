import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# SSL context for NCBI API (research/educational context)
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


PUBMED_SEARCH_CONFIG = {
    "diabetes": {
        "query": '(diabetes OR "type 2 diabetes" OR "glucose intolerance" OR "insulin resistance") AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
    "heart_disease": {
        "query": '(heart disease OR "ischemic heart" OR "cardiac risk" OR hypertension OR coronary) AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
    "pneumonia": {
        "query": '(pneumonia OR "respiratory infection" OR hypoxemia OR "oxygen saturation") AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
}

SAMPLE_ABSTRACTS = {
	"diabetes": [
		{
			"id": "sample_diabetes_1",
			"text": "Elevated glucose and obesity are strongly associated with type 2 diabetes severity and long-term complications.",
			"metadata": {"source": "Synthetic sample abstract", "domain": "diabetes"},
		},
		{
			"id": "sample_diabetes_2",
			"text": "Glycemic control and weight management remain central to reducing diabetes-related morbidity.",
			"metadata": {"source": "Synthetic sample abstract", "domain": "diabetes"},
		},
	],
	"heart_disease": [
		{
			"id": "sample_heart_1",
			"text": "Hypertension, dyslipidemia, and ischemic symptoms remain strong predictors of adverse cardiac outcomes.",
			"metadata": {"source": "Synthetic sample abstract", "domain": "heart_disease"},
		}
	],
	"pneumonia": [
		{
			"id": "sample_pneumonia_1",
			"text": "Hypoxemia, tachypnea, and inflammatory markers correlate with pneumonia severity and need for escalation.",
			"metadata": {"source": "Synthetic sample abstract", "domain": "pneumonia"},
		}
	],
}


def _fetch_xml(url: str) -> str:
	with urllib.request.urlopen(url, timeout=30, context=_SSL_CONTEXT) as response:
		return response.read().decode("utf-8", errors="ignore")


def _pubmed_search(term: str, limit: int = 100) -> list[str]:
	"""Search PubMed for article IDs matching the query term."""
	params = urllib.parse.urlencode({
		"db": "pubmed",
		"retmode": "json",
		"retmax": limit,
		"term": term,
		"sort": "relevance"
	})
	url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
	try:
		payload = json.loads(_fetch_xml(url))
		return payload.get("esearchresult", {}).get("idlist", [])
	except Exception as e:
		print(f"Error searching PubMed: {e}", file=sys.stderr)
		return []


def _pubmed_fetch_abstracts(pmid_list: list[str]) -> list[dict]:
	"""Fetch full article metadata including abstracts from PubMed IDs."""
	if not pmid_list:
		return []
	
	abstract_docs = []
	pmid_str = ",".join(pmid_list[:100])  # Limit to 100 per request
	params = urllib.parse.urlencode({
		"db": "pubmed",
		"id": pmid_str,
		"retmode": "xml",
		"rettype": "abstract"
	})
	url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
	
	try:
		xml_str = _fetch_xml(url)
		root = ET.fromstring(xml_str)
		
		for article in root.findall(".//PubmedArticle"):
			try:
				pmid_elem = article.find(".//PMID")
				pmid = pmid_elem.text if pmid_elem is not None else "unknown"
				
				title_elem = article.find(".//ArticleTitle")
				title = title_elem.text if title_elem is not None else ""
				
				abstract_elem = article.find(".//Abstract/AbstractText")
				abstract_text = abstract_elem.text if abstract_elem is not None else ""
				
				# Combine title + abstract for richer context
				full_text = f"{title}. {abstract_text}".strip() if abstract_text else title
				
				if full_text and len(full_text) > 50:  # Only keep substantive abstracts
					abstract_docs.append({
						"id": f"pubmed_{pmid}",
						"text": full_text[:2000],  # Cap at 2000 chars
						"metadata": {
							"source": f"PubMed:{pmid}",
							"pmid": pmid,
							"title": title,
						}
					})
			except Exception as e:
				print(f"Error parsing article: {e}", file=sys.stderr)
				continue
				
	except Exception as e:
		print(f"Error fetching PubMed abstracts: {e}", file=sys.stderr)
	
	return abstract_docs




def build_documents_for_domain(domain: str, limit: int = 30, allow_network: bool = True) -> list[dict]:
	"""Fetch and build documents for a specific domain."""
	config = PUBMED_SEARCH_CONFIG.get(domain)
	if not config:
		print(f"Domain {domain} not configured; using samples", file=sys.stderr)
		return SAMPLE_ABSTRACTS.get(domain, [])
	
	all_docs = SAMPLE_ABSTRACTS.get(domain, []).copy()  # Start with samples as fallback
	
	if allow_network:
		try:
			print(f"[info] Searching PubMed for {domain}...", file=sys.stderr)
			pmid_list = _pubmed_search(config["query"], limit=limit)
			
			if pmid_list:
				print(f"[info] Found {len(pmid_list)} articles; fetching abstracts...", file=sys.stderr)
				abstracts = _pubmed_fetch_abstracts(pmid_list)
				
				# Tag with domain
				for doc in abstracts:
					doc["metadata"]["domain"] = domain
				
				all_docs.extend(abstracts)
				print(f"[info] Successfully fetched {len(abstracts)} abstracts for {domain}", file=sys.stderr)
				
				# Rate limit to avoid hitting NCBI too hard
				time.sleep(1)
		except Exception as exc:
			print(f"[warn] PubMed fetch failed for {domain}: {exc}; using samples", file=sys.stderr)
	
	return all_docs


def main() -> int:
	parser = argparse.ArgumentParser(description="Fetch PubMed abstracts and prepare ChromaDB ingestion docs.")
	parser.add_argument("--domains", nargs="+", help="Domain keys (diabetes, heart_disease, pneumonia)")
	parser.add_argument("--limit", type=int, default=30, help="Max abstracts per domain")
	parser.add_argument("--offline", action="store_true", help="Use only sample data (no PubMed API calls)")
	parser.add_argument("--output", default="data/processed/pubmed_docs.json")
	args = parser.parse_args()

	targets = args.domains or list(PUBMED_SEARCH_CONFIG.keys())
	
	print(f"[info] Processing domains: {targets}", file=sys.stderr)
	all_docs: list[dict] = []
	
	for domain in targets:
		docs = build_documents_for_domain(domain, limit=args.limit, allow_network=not args.offline)
		all_docs.extend(docs)
		print(f"[info] Domain '{domain}': {len(docs)} documents", file=sys.stderr)

	out_path = Path(args.output)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(all_docs, indent=2), encoding="utf-8")
	
	print(f"\n✓ Wrote {len(all_docs)} total documents to {out_path}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
