import sys, io, os, json, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import session_persistence as sp

# ─── Test 1: Save and load session ───
print("=== Test 1: Save and Load Session ===")
prefs = {"distribution": 0.8, "correlation": 0.3, "missing_values": 0.5,
         "categorical": 0.5, "outliers": 0.5, "time_series": 0.5,
         "clustering": 0.5, "feature_importance": 0.5}
history = [
    {"recommendation_type": "distribution", "action": "liked", "timestamp": "2025-01-01T00:00:00"},
    {"recommendation_type": "correlation", "action": "disliked", "timestamp": "2025-01-01T00:01:00"}
]
ok = sp.save_session("test_session_1", prefs, history, active_goal="Explore distributions", last_file="data.csv")
assert ok is True
data = sp.load_session("test_session_1")
assert data is not None
assert data["preferences"] == prefs
assert data["active_goal"] == "Explore distributions"
assert data["last_file"] == "data.csv"
assert len(data["interaction_history"]) == 2
print("  Save/load round-trip works. Passed!")

# ─── Test 2: List sessions ───
print("\n=== Test 2: List Sessions ===")
sessions = sp.list_sessions()
ids = [s["id"] for s in sessions]
assert "test_session_1" in ids
print(f"  Listed {len(sessions)} session(s). Passed!")

# ─── Test 3: Overwrite session ───
print("\n=== Test 3: Overwrite Session ===")
prefs2 = prefs.copy()
prefs2["distribution"] = 0.9
ok = sp.save_session("test_session_1", prefs2, [], active_goal="Find relationships")
assert ok is True
data2 = sp.load_session("test_session_1")
assert data2["preferences"]["distribution"] == 0.9
assert data2["active_goal"] == "Find relationships"
print("  Overwrite works. Passed!")

# ─── Test 4: Load non-existent session ───
print("\n=== Test 4: Load Non-existent Session ===")
data3 = sp.load_session("nonexistent_session_xyz")
assert data3 is None
print("  Returns None for non-existent session. Passed!")

# ─── Test 5: Delete session ───
print("\n=== Test 5: Delete Session ===")
sp.save_session("test_delete_me", prefs, [])
deleted = sp.delete_session("test_delete_me")
assert deleted is True
data4 = sp.load_session("test_delete_me")
assert data4 is None
print("  Delete works. Passed!")

# ─── Test 6: Save with profile_json ───
print("\n=== Test 6: Save with profile_json ===")
profile = {"shape": [100, 5], "numerical_cols": ["a", "b"]}
ok = sp.save_session("test_profile", prefs, [], profile_json=profile)
assert ok is True
data5 = sp.load_session("test_profile")
assert "profile_json" in data5
assert data5["profile_json"]["shape"] == [100, 5]
print("  Profile JSON round-trip works. Passed!")

# ─── Test 7: Empty history ───
print("\n=== Test 7: Empty History ===")
ok = sp.save_session("test_empty_hist", prefs, [])
assert ok is True
data6 = sp.load_session("test_empty_hist")
assert data6["interaction_history"] == []
print("  Empty history handled. Passed!")

# ─── Cleanup test sessions ───
for sid in ["test_session_1", "test_profile", "test_empty_hist"]:
    sp.delete_session(sid)

print("\n=== ALL SESSION PERSISTENCE TESTS PASSED ===")
