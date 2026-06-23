from __future__ import annotations
import pickle 
from pathlib import Path    
from typing import Any, Optional
import pandas as pd

class DatabaseRepository:
    def __init__(self,database_dir: Path):
        self.database_dir=database_dir
        self.acc_dataset_dict =None
        self.acc_article_dict= None
        self.doi_dataset_dict= None
        self.doi_article_dict= None
        self.datacite_mapping=None
        self.crossref_mapping=None  
        self.acc_mapping=None
    def load(self) ->"DatabaseRepository":   
        with open(self.database_dir/"mdc_dataset.pkl","rb") as f:
            full_database=pickle.load(f)
        self.acc_dataset_dict=full_database["acc_dataset_dict"]
        self.acc_article_dict= full_database["acc_article_dict"]
        self.doi_dataset_dict= full_database["doi_dataset_dict"]
        self.doi_article_dict= full_database["doi_article_dict"]
        with open(self.database_dir/"datacite_mapping.pkl","rb") as f:
            self.datacite_mapping=pickle.load(f)
        with open(self.database_dir/"crossref_mapping.pkl","rb") as f:
            self.crossref_mapping=pickle.load(f)
        with open(self.database_dir/"samn_mapping.pkl","rb") as f:
            self.acc_mapping=pickle.load(f)
        return self
class LabelRepository:
    def __init__(self,label_path:Optional[Path]):
        self.label_path=label_path
    def load(self)->Optional[pd.DataFrame]:
        if self.label_path is None or not self.label_path.exists():
            return None
        return pd.read_csv(self.label_path)
# def main():
#     db_dir = Path("/mnt/sdb2/Hau/mdc_project/data/ExternalData/mdc_datasetv3")
#     label_path = Path("/mnt/sdb2/Hau/mdc_project/data/InternalData/make-data-count-finding-data-references/train_labels.csv")

#     db_repo = DatabaseRepository(db_dir)
#     label_repo = LabelRepository(label_path)

#     db_repo.load()
#     labels = label_repo.load()

#     print("DB loaded")
#     print("Labels:", None if labels is None else labels.shape)

# if __name__ == "__main__":
#     main()