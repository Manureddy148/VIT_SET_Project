"""Image-only CNN: never invent vital signs from pixels or encoded bytes."""
from io import BytesIO
import torch
from PIL import Image
from torchvision import transforms
from src.models.base_model import PredictionResult

def network():
    return torch.nn.Sequential(torch.nn.Conv2d(1,16,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),
       torch.nn.Conv2d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(32,1))
TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.5],[.5])])

class PneumoniaImageModel:
    def __init__(self,path):
        b=torch.load(path,map_location='cpu',weights_only=True)
        if b.get('synthetic',True):raise ValueError('Synthetic image checkpoint is not allowed for inference')
        self.model=network();self.model.load_state_dict(b['state_dict']);self.model.eval()
        self.platt=b['platt'];self.metadata=b['metadata']
    def predict_bytes(self,payload):
        if len(payload)>20*1024*1024:raise ValueError('Image exceeds 20MB')
        image=Image.open(BytesIO(payload));image.verify()
        image=Image.open(BytesIO(payload)).convert('L')
        if image.width*image.height>25_000_000:raise ValueError('Image dimensions too large')
        tensor=TRANSFORM(image).unsqueeze(0)
        with torch.no_grad(): logit=float(self.model(tensor).item())
        p=float(torch.sigmoid(torch.tensor(self.platt[0]*logit+self.platt[1])).item())
        score=round(100*p,1)
        # No fake SHAP values: image explanations require separate attribution validation.
        label='Low' if score<25 else 'Moderate' if score<50 else 'High' if score<75 else 'Critical'
        return PredictionResult('pneumonia',score,max(p,1-p),label,{},['chest_xray'],[])
