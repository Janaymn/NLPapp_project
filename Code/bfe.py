import torch
import re

class BFE:
    """
    Behavioral Feature Extractor (BFE)
    Encodes the student's behavior and profile into a state vector for the DQN.
    """
    def __init__(self):
        # Keywords associated with confusion or struggle
        self.confusion_keywords = [
            "i don't understand", "confused", "not sure", "hard",
            "difficult", "why", "how come", "wrong", "mistake"
        ]

    def _analyze_confusion(self, text):
        if not text:
            return 1.0
        text = text.lower()
        matches = sum(1 for word in self.confusion_keywords if word in text)
        return min(1.0, matches / 3.0) # Normalize to [0, 1]

    def encode(self, raw_response, profile, history=None, knowledge_code=None, kcg=None):
        """
        Convert behavior and profile to a state vector.
        Now includes the Learning Objectives Vector (g_t) as per the methodology.
        """
        # 1. Behavioral Feature Vector (f_t)
        confusion = self._analyze_confusion(raw_response)
        activity = 1.0 if profile.activity() == 'high' else 0.0
        diversity = 1.0 if profile.diversity() == 'high' else 0.0
        ability_map = {'good': 1.0, 'common': 0.5, 'poor': 0.0}
        ability = ability_map.get(profile.ability(), 0.0)
        length = min(1.0, len(raw_response) / 200.0) if raw_response else 0.0

        history_score = 0.5
        if history and len(history) > 0:
            recent_corr = [h.get('corr', 0) for h in history[-3:]]
            history_score = sum(recent_corr) / len(recent_corr)

        f_t = torch.FloatTensor([confusion, activity, diversity, ability, length, history_score])

        # 2. Learning Objectives Vector (g_t)
        # We represent g_t as a normalized knowledge_code relative to the KCG size
        # or a one-hot vector. To keep dimensionality manageable, we use a normalized index.
        if knowledge_code is not None and kcg is not None:
            # Normalize knowledge_code by the total number of concepts in KCG
            g_t = torch.FloatTensor([knowledge_code / max(len(kcg), 1)])
        else:
            g_t = torch.FloatTensor([0.0])

        # Combine into a state vector s_t = (f_t, g_t)
        return torch.cat([f_t, g_t])
