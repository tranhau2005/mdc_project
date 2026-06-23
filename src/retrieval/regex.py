from __future__ import annotations
import re

class RegexRegistry:
    def __init__(self):
        self.doi_start=re.compile(r"1\s*0\s*\.")
        self.accession=re.compile(
            r"(?:"
            r"EPI_ISL_\d{5,}|EPI\d{5,}|"
            r"G(?:PL|SM|SE|DS)\d+|"
            r"(?:(?:WP|AC|AP|NC|NG|NM|NP|NR|NT|NW|XM|XP|XR|YP|ZP)_\d+|NZ_[A-Z]{2,4}\d+)(?:\.\d+)?|"
            r"(?:[A-NR-Z][0-9](?:[A-Z][A-Z0-9][A-Z0-9][0-9]){1,2}|[OPQ][0-9][A-Z0-9][A-Z0-9][A-Z0-9][0-9])(?:\.\d+)?|"
            r"PRJ[DEN][A-Z]\d+|"
            r"(?:ENS[FPTG]\d{11}(?:\.\d+)?|FB\w{2}\d{7}|Y[A-Z]{2}\d{3}[a-zA-Z](?:-[A-Z])?|[A-Z_a-z0-9]+(?:\.)?(?:t)?(?:\d+)?(?:[a-z])?)|"
            r"PF\d{5}|"
            r"[AEP]-\w{4}-\d+|"
            r"EMD-\d{4,5}|"
            r"IPR\d{6}|"
            r"phs[0-9]{6}(?:\.v\d+\.p\d+)?|"
            r"S-[A-Z]{4}[\-_A-Z\d]+|"
            r"[1-6]\.[0-9]+\.[0-9]+\.[0-9]+|"
            r"CPX-[0-9]+|"
            r"(?:R-[A-Z]{3}-\d+(?:-\d+)?(?:\.\d+)?|REACT_\d+(?:\.\d+)?)|"
            r"GO:\d{7}|"
            r"EBI-[0-9]+|"
            r"UPI[A-F0-9]{10}|"
            r"URS[0-9A-F]{10}(?:_\d+)?|"
            r"ENSG\d{11}|"
            r"(?:(?:BIOMD|MODEL)\d{10}|BMID\d{12})|"
            r"CHEMBL\d+|"
            r"EGAD\d{11}|"
            r"MTBLS\d+|"
            r"EMPIAR-\d{5,}|"
            r"RF\d{5}|"
            r"\w{1,2}\d+|"
            r"[0-9][A-Za-z0-9]{3}|"
            r"[A-Z]+[0-9]+(?:\.\d+)?"
            r")"
        )