"""Validated modality identifiers and actual input decoders.
Models are separately trained/registered. No derived cholesterol/glucose/vitals.
"""
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
MODALITIES=('text','tabular','xray','ct','mri','ultrasound','echo','dermatology','fundus','oct','pathology','ecg','audio','genomics')
def route_modality(modality):
    if modality not in MODALITIES:raise ValueError(f'Unknown modality: {modality}')
    return modality

def ecg_features(signal,fs):
    x=np.asarray(signal,dtype=float)
    if fs<=80 or x.ndim!=1 or len(x)<fs*5 or not np.isfinite(x).all():raise ValueError('Finite 1D ECG >=5 seconds, sampling rate >80Hz required')
    filtered=sosfiltfilt(butter(3,[.5,40],fs=fs,btype='bandpass',output='sos'),x)
    peaks,_=find_peaks(filtered,distance=int(fs*.3),prominence=max(filtered.std()*.6,1e-8))
    if len(peaks)<3:raise ValueError('Insufficient detected beats')
    rr=np.diff(peaks)/fs*1000
    return {'r_peaks':peaks.tolist(),'mean_hr':float(60000/rr.mean()),'sdnn_ms':float(rr.std(ddof=1)),
            'rmssd_ms':float(np.sqrt(np.mean(np.diff(rr)**2))),'pnn50':float(np.mean(np.abs(np.diff(rr))>50)),
            'warning':'Signal features only; QRS/QT/ST and rhythm diagnosis require a validated delineator/model.'}

def audio_features(path):
    import soundfile as sf
    x,fs=sf.read(path,always_2d=True)
    x=x.mean(axis=1)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Empty/nonfinite decoded audio')
    return {'sample_rate':fs,'duration_seconds':len(x)/fs,'rms':float(np.sqrt(np.mean(x*x))),
            'zero_crossing_rate':float(np.mean(np.diff(np.sign(x))!=0)),'diagnosis':None}

def vcf_variants(path):
    rows=[]
    with open(path) as f:
        for line in f:
            if line.startswith('#'):continue
            cells=line.rstrip().split('\t')
            if len(cells)<8:raise ValueError('Invalid VCF record')
            rows.append({'chrom':cells[0],'position':int(cells[1]),'ref':cells[3],'alt':cells[4],
                         'filter':cells[6],'info':cells[7]})
    return rows  # no glucose/diabetes conclusions from variant words

def load_volume(path):
    from pathlib import Path
    if str(path).endswith(('.nii','.nii.gz')):
        import nibabel as nib
        img=nib.load(path);return np.asanyarray(img.dataobj),img.affine
    import pydicom
    ds=pydicom.dcmread(path)
    return ds.pixel_array,None
