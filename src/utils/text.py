from __future__ import annotations
import ast 
import re
import unicodedata
from nameparser import HumanName
from rapidfuzz import fuzz 
def normalize(text:str)->str:
    text=unicodedata.normalize("NFKC",text or "")
    return re.sub(r"[^\x00-\x7F]","",text)
def clean_title(title:str)->str:
    title=normalize(title or "")
    title=title.replace("[", "").replace("]", "")
    title=title.replace("<i>", "").replace("</i>", "")
    title=title.replace("\n","")
    title=title.replace("Data from:","")
    title=title.replace("Occurrence Download","")
    return title.strip()
def clean_doi(doi:str | None)-> str | None:
    doi = (doi or "").lower()
    for token in [
         " ", "\n", "\t", "\r",
        "https://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/",
        "http://orcid.org/", "doi.org/"
    ]:
        doi = doi.replace(token, "")
    doi = doi.strip()
    if not doi.startswith("10."):
        return None
    part=doi.split("/")
    if len(part)<2:
        return None
    prefix=part[0]
    suffix="/".join(part[1:])
    if len(suffix)<2:
        return None
    return f"{prefix}/{suffix}"

def clean_doi_alpha_num(text:str | None)-> str | None:
    cleaned=clean_doi(text) or ""
    return "".join(ch for ch in cleaned if ch.isalnum())
def clean_doi_alpha_num_article_id(text:str | None )-> str :
    text =text or ""
    if "_" in text:
        text= text.split("_")[0]+"/"+"_".join(text.split("_")[1:])
    return clean_doi_alpha_num(text)
def clean_article_id(article_doi: str, split:bool=False)-> str:
    if split and "_" in article_doi:
        article_doi=article_doi.split("_")[0]+"/"+"_".join(article_doi.split("_")[1:])
    return article_doi.replace("/","").replace("_","").replace(".","").lower()
def eval_string_name(name):
    if isinstance(name, list):
        return name
    try:
        return ast.literal_eval(name)
    except Exception:
        return str(name).replace("[","").replace("]","").split(",")
def has_authors_match(dataset_authors, article_authors)-> bool:
    def parse_author(author: str):
        try:
            name = HumanName(author)
        except Exception:
            return ("","")
        last_name=name.last.lower()
        first_name=name.first.lower()
        return (last_name, first_name)
    dataset_authors=eval_string_name(dataset_authors)
    article_authors=eval_string_name(article_authors)
    dataset_parsed={parse_author(author) for author in dataset_authors}
    article_parsed={parse_author(author) for author in article_authors}
    dataset_parsed={x for x in dataset_parsed if x[0] and  x[1]}
    article_parsed={x for x in article_parsed if x[0] and  x[1]}
    return len(dataset_parsed & article_parsed)>0
def get_title_dist(dataset_title,article_title)-> int:
    if isinstance(dataset_title, list) and  dataset_title:
        dataset_title=" ".join(dataset_title)
    if isinstance(article_title, list) and article_title:
        article_title=" ".join(article_title)
    dataset_title= clean_title(dataset_title or "").lower()
    article_title= clean_title(article_title or "").lower()
    if len(dataset_title)<10 or len(article_title)<10:
        return 0        
    return fuzz.WRatio(dataset_title, article_title)
