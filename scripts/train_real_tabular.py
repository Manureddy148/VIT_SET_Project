"""Train on UCI data, with stratified train/calibration/test and no test tuning."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from src.training.metrics import classification_metrics

SOURCES = {
 'heart_disease': ('https://archive.ics.uci.edu/static/public/45/data.csv', 'num'),
 'diabetes': ('https://archive.ics.uci.edu/static/public/891/data.csv', 'Diabetes_binary'),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=SOURCES, required=True)
    parser.add_argument('--csv', help='Local copy; downloads official UCI CSV if omitted')
    parser.add_argument('--max-rows', type=int, default=0, help='0=all; subset results must not be called full-data results')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()
    url, target = SOURCES[args.domain]
    raw = Path(args.csv or f'data/raw/{args.domain}.csv')
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        import urllib.request
        urllib.request.urlretrieve(url, raw)
    df = pd.read_csv(raw)
    df = df.dropna(subset=[target])
    if args.max_rows and args.max_rows < len(df):
        _, df = train_test_split(df, test_size=args.max_rows, stratify=(df[target] > 0), random_state=42)
    y = (df[target] > 0).astype(int)
    X = df.drop(columns=[target, 'ID', 'id'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    Xtrain, Xcal, ytrain, ycal = train_test_split(Xdev, ydev, test_size=.25, stratify=ydev, random_state=43)
    base = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('model', XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
                                             random_state=42, n_jobs=2, eval_metric='logloss'))])
    base.fit(Xtrain, ytrain)
    # sklearn >=1.6 FrozenEstimator protects fitted training model from refitting.
    from sklearn.frozen import FrozenEstimator
    model = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    model.fit(Xcal, ycal)
    uncal, calibrated = base.predict_proba(Xtest)[:,1], model.predict_proba(Xtest)[:,1]
    metadata = {'domain': args.domain, 'source': url, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                'rows': len(df), 'train_rows': len(Xtrain), 'calibration_rows': len(Xcal), 'test_rows':len(Xtest),
                'subset': bool(args.max_rows), 'seed':42, 'purpose':'research disease classification, NOT clinical severity'}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline':model, 'features':list(X.columns), 'metadata':metadata}, out/f'{args.domain}_research.joblib')
    report = {'metadata':metadata, 'uncalibrated':classification_metrics(ytest, uncal),
              'platt_calibrated':classification_metrics(ytest, calibrated), 'target_auc':.85,
              'target_met': bool(classification_metrics(ytest, calibrated)['auc'] > .85)}
    Path('reports').mkdir(exist_ok=True)
    Path(f'reports/{args.domain}_metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
