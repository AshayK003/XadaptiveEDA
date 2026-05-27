import json
import os

import pandas as pd

from constants import DEFAULT_PREFERENCES, ANALYSIS_GOALS, DECAY_HALF_LIFE_HOURS


class PreferenceTracker:
    """Tracks user feedback and adjusts priority weights within [0.1, 1.0]."""

    ADJUSTMENTS = {
        'liked': 0.10,
        'disliked': -0.10,
        'explored': 0.05,
        'column_selected': 0.03,
        'viz_type_changed': 0.03,
        'ignored': -0.02,
    }

    def __init__(self):
        self.preference_weights = DEFAULT_PREFERENCES.copy()
        self.interaction_history = []
        self.active_goal = None
        self.goal_overrides_active = False

    def track_interaction(self, recommendation, action, timestamp=None, metadata=None):
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        entry = {
            'recommendation_type': recommendation['type'],
            'recommendation_title': recommendation['title'],
            'action': action,
            'timestamp': timestamp,
            'score': recommendation.get('score', 0),
        }
        if metadata:
            entry['metadata'] = metadata

        self.interaction_history.append(entry)
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

    def set_goal(self, goal_name):
        if goal_name is None:
            self.active_goal = None
            self.goal_overrides_active = False
            return
        if goal_name not in ANALYSIS_GOALS:
            raise ValueError(f"Unknown goal '{goal_name}'. Available: {list(ANALYSIS_GOALS.keys())}")
        self.active_goal = goal_name
        self.goal_overrides_active = True
        goal_weights = ANALYSIS_GOALS[goal_name]
        self.preference_weights.clear()
        self.preference_weights.update(goal_weights)
        self.interaction_history.append({
            'recommendation_type': 'goal_set',
            'recommendation_title': f"Goal: {goal_name}",
            'action': 'goal_set',
            'timestamp': pd.Timestamp.now(),
            'metadata': {'goal': goal_name}
        })

    def apply_temporal_decay(self, half_life_hours=DECAY_HALF_LIFE_HOURS):
        if not self.interaction_history:
            return
        now = pd.Timestamp.now()
        for entry in self.interaction_history:
            ts = entry.get('timestamp')
            if ts is None:
                continue
            delta_hours = (now - pd.Timestamp(ts)).total_seconds() / 3600
            if delta_hours <= 0:
                continue
            decay_factor = 2 ** (-delta_hours / half_life_hours)
            rec_type = entry['recommendation_type']
            if rec_type in self.preference_weights:
                current = self.preference_weights[rec_type]
                self.preference_weights[rec_type] = max(0.1, min(1.0, 0.5 + (current - 0.5) * decay_factor))
        self.interaction_history.append({
            'recommendation_type': 'decay_applied',
            'recommendation_title': 'Temporal Decay',
            'action': 'decay_applied',
            'timestamp': now
        })

    def save_preferences(self, path=None):
        if path is None:
            path = os.path.expanduser("~/.eda_assistant_prefs.json")
        data = {
            'preference_weights': self.preference_weights.copy(),
            'active_goal': self.active_goal,
            'goal_overrides_active': self.goal_overrides_active,
            'interaction_count': len(self.interaction_history)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_preferences(self, path=None):
        if path is None:
            path = os.path.expanduser("~/.eda_assistant_prefs.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.preference_weights.clear()
            self.preference_weights.update(data.get('preference_weights', DEFAULT_PREFERENCES))
            self.active_goal = data.get('active_goal')
            self.goal_overrides_active = data.get('goal_overrides_active', False)
            return True
        except (json.JSONDecodeError, OSError):
            return False 