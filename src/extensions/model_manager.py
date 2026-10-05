"""Atomic hot-swap of fully initialized compatible models, with version audit."""
from threading import RLock
from datetime import datetime,timezone
class ModelManager:
    def __init__(self):self._lock=RLock();self._models={};self.events=[]
    def replace(self,name,model,version):
        if not version or not callable(getattr(model,'predict',None)):raise ValueError('Model must implement predict and declare version')
        with self._lock:
            old=self._models.get(name)
            self._models[name]=(model,version)
            self.events.append({'name':name,'version':version,'previous':old[1] if old else None,'at':datetime.now(timezone.utc).isoformat()})
    def predict(self,name,features):
        with self._lock:model,version=self._models[name]
        return model.predict(features),version
    def versions(self):
        with self._lock:return {k:v[1] for k,v in self._models.items()}
