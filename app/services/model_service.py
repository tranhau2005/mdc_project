from __future__ import annotations
import asyncio 
from pathlib import Path
import torch
from src.classification.deberta_classifier import DebertaClassifier
from src.classification.ensemble import EnsembleClassifier
from src.config import AppConfig,load_config
from src.utils.gpu import configure_torch, seed_everything

class ModelService: 
    def __init__(self,config_path:str)-> None:
        self.config_path=Path(config_path)
        self.config: AppConfig | None=None
        self.ensemble: EnsembleClassifier | None = None
        self.is_ready = False
        self.loading_error:str | None =None
        # Just permit only 1 inference session run on gpu 
        self.inference_semaphore= asyncio.Semaphore(1)
    def load(self)-> None:
        """
        Loading config and model. This function is blocking
        """
        if self.is_ready:
            return
        try: 
            config = load_config(self.config_path)
            seed_everything(config.runtime.seed)
            configure_torch()
            classifiers: list[DebertaClassifier]=[]
            for model_path in config.model.model_paths: 
                path=Path(model_path)
                if not path.exists(): 
                    raise FileNotFoundError(
                        f"Model directory does not exist. {path}"
                    )
                classifier = DebertaClassifier(
                    model_path= model_path,
                    tokenizer_path=config.model.tokenizer_path,
                    max_length=config.model.max_length,
                    batch_size=config.model.batch_size,
                    device=config.model.device,
                    use_fp16=config.model.use_fp16
                )
                classifier.load()
                classifiers.append(classifier)
            if not classifiers: 
                raise RuntimeError("No mode was loaded")
            self.config =config
            self.ensemble = EnsembleClassifier(classifiers= classifiers)
            self.loading_error = None
            self.is_ready= True
        except Exception as exc:
            self.is_ready=False 
            self.loading_error= str(exc)
            raise
          
    def close(self): 
        """
        Free model and GPU when the application is used
        """
        self.is_ready=False
        self.ensemble = None
        self.config=None
        
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        