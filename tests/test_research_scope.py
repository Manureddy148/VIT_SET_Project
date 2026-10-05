import numpy as np
import pytest
from src.training.metrics import expected_calibration_error,retrieval_metrics
from src.extensions.modalities import route_modality,MODALITIES,ecg_features,vcf_variants
from src.extensions.model_manager import ModelManager
from src.extensions.dialogue import missing_data_dialogue
from src.rag.hallucination_guard import evaluate_faithfulness

def test_ece_boundaries():
    assert expected_calibration_error([0,1],[0,1])==0
    with pytest.raises(ValueError):expected_calibration_error([0],[1.1])
def test_retrieval_metrics():
    r=retrieval_metrics(['a','b','c'],{'a','c'},k=3)
    assert r['precision']==pytest.approx(2/3) and r['recall']==1
    with pytest.raises(ValueError):retrieval_metrics(['a','a'],{'a'})
def test_no_citation_no_faithfulness():assert evaluate_faithfulness('Any claim',[])[0] is False
def test_all_routes():
    assert len(MODALITIES)==14
    for m in MODALITIES:assert route_modality(m)==m
    with pytest.raises(ValueError):route_modality('made_up')
def test_ecg_real_signal():
    fs=250;t=np.arange(fs*10)/fs;x=sum(np.exp(-((t-i)/.012)**2) for i in range(1,10))
    r=ecg_features(x,fs)
    assert 55<r['mean_hr']<65
    with pytest.raises(ValueError):ecg_features([1,2],fs)
def test_vcf(tmp_path):
    p=tmp_path/'test.vcf';p.write_text('##fileformat=VCFv4.2\n1\t123\t.\tA\tC\t.\tPASS\t.\n')
    assert vcf_variants(p)[0]['position']==123
def test_hotswap():
    class Model:
        def __init__(self,v):self.v=v
        def predict(self,x):return x+self.v
    m=ModelManager();m.replace('test',Model(1),'v1');assert m.predict('test',2)==(3,'v1')
    m.replace('test',Model(4),'v2');assert m.predict('test',2)==(6,'v2')
def test_dialogue():assert missing_data_dialogue(['age','BMI'],{'age':42})[0]['field']=='BMI'
def test_image_manifest_patient_leak():
    pytest.importorskip("torch")
    import pandas as pd
    from scripts.train_pneumonia_image import validate_manifest
    df=pd.DataFrame({'path':['a','b'],'label':[0,1],'patient_id':['same','same'],'split':['train','test']})
    with pytest.raises(ValueError,match='Patient leakage'):validate_manifest(df)
def test_image_network_forward():
    torch = pytest.importorskip("torch")
    from src.models.pneumonia_image import network
    assert network()(torch.zeros(2,1,224,224)).shape==(2,1)

def test_no_artifact_no_synthetic_model(monkeypatch,tmp_path):
    from src.api.bootstrap import build_registry
    monkeypatch.setenv('MEDICAL_AI_ALLOW_DEMO','0');monkeypatch.setenv('MODEL_REGISTRY_DIR',str(tmp_path))
    assert build_registry().list_domains()==[]

def test_proxy_refused_by_default(monkeypatch):
    from src.preprocessing.multimodal_router import image_upload_to_clinical_features
    monkeypatch.setenv('MEDICAL_AI_ALLOW_DEMO','0')
    with pytest.raises(ValueError,match='Synthetic proxy'):
        image_upload_to_clinical_features('x.jpg',b'bytes','',{})

def test_dose_guard():
    assert evaluate_faithfulness('Take 5 mg daily',[{'url':'https://example.test','text':'source'}])[0] is False
