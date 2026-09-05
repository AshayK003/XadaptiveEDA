import io
import json
import os
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

from preference_learner import PreferenceTracker

# ─── Test 1: Goal setting ───
print("=== Test 1: Goal Setting ===")
tracker = PreferenceTracker()
tracker.set_goal('Explore distributions')
assert tracker.active_goal == 'Explore distributions'
assert tracker.goal_overrides_active
assert tracker.preference_weights['distribution'] == 0.9
assert tracker.preference_weights['missing_values'] == 0.3
# verify log entry
goal_entries = [e for e in tracker.interaction_history if e['action'] == 'goal_set']
assert len(goal_entries) == 1
print("  Goal correctly overrides weights. Passed!")

# ─── Test 2: Clear goal ───
print("\n=== Test 2: Clear Goal ===")
tracker.set_goal(None)
assert tracker.active_goal is None
assert not tracker.goal_overrides_active
# weights should still be at goal values (we don't auto-revert)
assert tracker.preference_weights['distribution'] == 0.9
print("  Goal cleared without side effects. Passed!")

# ─── Test 3: Unknown goal raises ───
print("\n=== Test 3: Unknown Goal ===")
try:
    tracker.set_goal('nonexistent')
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("  Unknown goal correctly raises ValueError. Passed!")

# ─── Test 4: temporal decay (no-op on empty history) ───
print("\n=== Test 4: Temporal Decay Empty ===")
tracker2 = PreferenceTracker()
weights_before = tracker2.preference_weights.copy()
tracker2.apply_temporal_decay()
assert tracker2.preference_weights == weights_before
print("  No-op on empty history. Passed!")

# ─── Test 5: temporal decay shifts weights toward 0.5 ───
print("\n=== Test 5: Temporal Decay Effect ===")
tracker3 = PreferenceTracker()
tracker3.preference_weights['distribution'] = 0.9
tracker3.preference_weights['missing_values'] = 0.2
# Simulate old interaction (12 hours ago)
old_ts = pd.Timestamp.now() - pd.Timedelta(hours=12)
tracker3.track_interaction({'type': 'distribution', 'title': 'D', 'score': 0.8}, 'liked', timestamp=old_ts)
tracker3.apply_temporal_decay(half_life_hours=4.0)  # 12h = 3 half-lives -> factor ~0.125
assert tracker3.preference_weights['distribution'] < 0.9, "Should decay toward 0.5"
assert tracker3.preference_weights['distribution'] >= 0.1
decay_entries = [e for e in tracker3.interaction_history if e['action'] == 'decay_applied']
assert len(decay_entries) == 1
print(f"  Weight decayed from 0.9 to {tracker3.preference_weights['distribution']:.2f}. Passed!")

# ─── Test 6: Save/Load preferences ───
print("\n=== Test 6: Save/Load Preferences ===")
tracker4 = PreferenceTracker()
tracker4.set_goal('Find relationships')
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    save_path = f.name

tracker4.save_preferences(save_path)

with open(save_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
assert 'preference_weights' in data
assert data['active_goal'] == 'Find relationships'
assert data['goal_overrides_active']
assert data['preference_weights']['correlation'] == 0.9

# Load into a fresh tracker
tracker5 = PreferenceTracker()
ok = tracker5.load_preferences(save_path)
assert ok
assert tracker5.preference_weights['correlation'] == 0.9
assert tracker5.active_goal == 'Find relationships'
assert tracker5.goal_overrides_active

os.unlink(save_path)
print("  Save/Load round-trip preserves goal and weights. Passed!")

# ─── Test 7: Load non-existent file returns False ───
print("\n=== Test 7: Load Non-existent ===")
tracker6 = PreferenceTracker()
ok = tracker6.load_preferences('/nonexistent/path.json')
assert not ok
print("  Non-existent file returns False. Passed!")

print("\n=== ALL PHASE 3 TESTS PASSED ===")
