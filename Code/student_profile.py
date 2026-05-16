import json
from config import DATA_PATH

class Profile:
    """
    Learner Profile Module (§4.1.4.2)
    提取 explicit practice styles 和 implicit ability
    """
    def __init__(self, agent_id, profile_data=None):
        if profile_data:
            # Use provided neurodiverse profile data
            self.values = [
                "id",
                profile_data.get("activity", "low"),
                profile_data.get("diversity", "low"),
                profile_data.get("preference", "unknown"),
                profile_data.get("success_rate", "low"),
                profile_data.get("ability", "poor")
            ]
        else:
            path = f"{DATA_PATH}/profile.json"
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            # raw: "id\tactivity\tdiversity\tpreference\tsuccess_rate\tabs"
            self.values = data[str(agent_id)].split('\t')


    def activity(self):
        val = self.values[1]
        try:
            mean = 0.0398856589
            return 'high' if float(val) > mean else 'low'
        except ValueError:
            return val

    def diversity(self):
        val = self.values[2]
        try:
            mean = 0.0627161572
            return 'high' if float(val) > mean else 'low'
        except ValueError:
            return val

    def preference(self):
        return self.values[3]

    def success_rate(self):
        val = self.values[4]
        try:
            ar = float(val)
            if ar > 0.6: return 'high'
            if ar > 0.3: return 'medium'
            return 'low'
        except ValueError:
            return val

    def ability(self):
        val = self.values[5]
        try:
            ab = float(val)
            if ab > 0.5: return 'good'
            if ab > 0.4: return 'common'
            return 'poor'
        except ValueError:
            return val

    def build_prompt(self):
        tips_act = {
            'high': 'you maintain a high level of online exercise activity and practice frequently',
            'low':  'you practice less regularly and with lower enthusiasm'
        }
        tips_div = {
            'high': 'you explore diverse knowledge categories',
            'low':  'you focus on limited knowledge categories'
        }
        return (
            f"You are a student with {self.activity()} activity, "
            f"{tips_act[self.activity()]}. "
            f"You have {self.diversity()} diversity, "
            f"{tips_div[self.diversity()]}. "
            f"Most practiced concept: {self.preference()}. "
            f"Success rate: {self.success_rate()}. "
            f"Problem-solving ability: {self.ability()}."
        )
        