"""CNN train/calibration/test using manifest with patient-disjoint splits.
Manifest columns: path,label,patient_id,split. label=0 NORMAL,1 PNEUMONIA.
Never infer patient IDs for normal scans from filenames without documented mapping.
"""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from sklearn.linear_model import LogisticRegression
from src.models.pneumonia_image import network,TRANSFORM
from src.training.metrics import classification_metrics
class Scans(Dataset):
    def __init__(self,df):self.df=df.reset_index(drop=True)
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];return TRANSFORM(Image.open(r.path).convert('L')),torch.tensor(float(r.label))
def validate_manifest(df):
    if not {'path','label','patient_id','split'}.issubset(df):raise ValueError('Missing manifest columns')
    if df.patient_id.isna().any() or df.path.duplicated().any():raise ValueError('Missing patient IDs or duplicate files')
    if not set(df.label).issubset({0,1}):raise ValueError('Binary labels required')
    if (df.groupby('patient_id').split.nunique()>1).any():raise ValueError('Patient leakage across splits')
    for split in ['train','calibration','test']:
        if df[df.split==split].label.nunique()!=2:raise ValueError('Both classes required in each split')
    # Hash leakage check catches copied files under different names.
    hashes={}
    for r in df.itertuples():
        h=hashlib.sha256(Path(r.path).read_bytes()).hexdigest()
        if h in hashes and hashes[h]!=r.split:raise ValueError('Duplicate image content across splits')
        hashes[h]=r.split

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',default='models/pneumonia_image.pt');a=p.parse_args()
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    df=pd.read_csv(a.manifest);validate_manifest(df)
    model=network();optimizer=torch.optim.Adam(model.parameters(),lr=.001);loss_fn=torch.nn.BCEWithLogitsLoss()
    loader=DataLoader(Scans(df[df.split=='train']),batch_size=16,shuffle=True)
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,y in loader:
            optimizer.zero_grad();loss=loss_fn(model(x).flatten(),y);loss.backward();optimizer.step();losses.append(loss.item())
        print(json.dumps({'epoch':epoch+1,'train_loss':float(np.mean(losses))}))
    def logits(split):
        rows=df[df.split==split];outputs=[];model.eval()
        with torch.no_grad():
            for x,_ in DataLoader(Scans(rows),batch_size=16):outputs.extend(model(x).flatten().tolist())
        return np.array(outputs),rows.label.to_numpy()
    zcal,ycal=logits('calibration');platt=LogisticRegression().fit(zcal.reshape(-1,1),ycal)
    ztest,ytest=logits('test');prob=platt.predict_proba(ztest.reshape(-1,1))[:,1]
    metadata={'rows':len(df),'split_counts':df.split.value_counts().to_dict(),'epochs':a.epochs,'patient_disjoint':True,'seed':42,'clinical_validation':False}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'platt':[float(platt.coef_[0,0]),float(platt.intercept_[0])],'metadata':metadata,'synthetic':False},out)
    report={'metadata':metadata,'platt_calibrated':classification_metrics(ytest,prob)}
    Path('reports').mkdir(exist_ok=True);Path('reports/pneumonia_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
