import pandas as pd

from constants import DEFAULT_PREFERENCES


class PreferenceTracker:
    """Tracks user feedback and adjusts priority weights within [0.1, 1.0]."""

    ADJUSTMENTS = {
        'liked': 0.10,
        'disliked': -0.10,
    }

    def __init__(self):
        self.preference_weights = DEFAULT_PREFERENCES.copy()
        self.interaction_history = []

    def track_interaction(self, recommendation, action, timestamp=None):
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        self.interaction_history.append({
            'recommendation_type': recommendation['type'],
            'recommendation_title': recommendation['title'],
            'action': action,
            'timestamp': timestamp,
            'score': recommendation.get('score', 0)
        })

        return self._apply_adjustment(recommendation['type'], action)

    def _apply_adjustment(self, rec_type, action):
        delta = self.ADJUSTMENTS.get(action, 0)
        if delta == 0:
            return self.preference_weights
        current = self.preference_weights.get(rec_type, 0.5)
        self.preference_weights[rec_type] = max(0.1, min(1.0, current + delta))
        return self.preference_weights

    def get_preferences(self):
        return self.preference_weights

    def get_top_preferences(self, n=3):
        sorted_prefs = sorted(self.preference_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_prefs[:n]

    def get_interaction_stats(self):
        if not self.interaction_history:
            return {"total_interactions": 0}
        stats = {"total_interactions": len(self.interaction_history), "action_counts": {}, "type_counts": {}}
        for interaction in self.interaction_history:
            action = interaction['action']
            rec_type = interaction['recommendation_type']
            stats["action_counts"][action] = stats["action_counts"].get(action, 0) + 1
            stats["type_counts"][rec_type] = stats["type_counts"].get(rec_type, 0) + 1
        return stats

    def set_preferences(self, preferences):
        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be a dictionary")
        required_keys = set(self.preference_weights.keys())
        provided_keys = set(preferences.keys())
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(f"Missing preference keys: {missing}")
        for key, value in preferences.items():
            if key in required_keys and (value < 0.1 or value > 1.0):
                raise ValueError(f"Preference value for {key} must be between 0.1 and 1.0")
        for key in required_keys:
            if key in preferences:
                self.preference_weights[key] = preferences[key]
        self.interaction_history.append({
            'recommendation_type': 'manual_adjustment',
            'recommendation_title': 'Manual Preference Update',
            'action': 'manual_set',
            'timestamp': pd.Timestamp.now()
        })
        return self.preference_weights 