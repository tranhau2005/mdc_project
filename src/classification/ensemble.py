from __future__ import annotations

import numpy as np


class EnsembleClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict_proba(self, prompts):
        if not self.classifiers:
            return np.empty((0, 2))
        all_probs = [clf.predict_proba(prompts) for clf in self.classifiers]
        return np.mean(np.stack(all_probs), axis=0)
