import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from llm_adapter import (
    RATE_LIMIT_CALLS,
    RATE_LIMIT_WINDOW,
    _check_rate_limit,
    _remote_call_log,
)

# ─── Test 1: Local provider always allowed ───
print("=== Test 1: Local Provider Unrestricted ===")
assert _check_rate_limit("local") is True
assert _check_rate_limit("local") is True
print("  Local provider never rate-limited. Passed!")

# ─── Test 2: Remote provider within limit ───
print("\n=== Test 2: Remote Within Limit ===")
_remote_call_log.clear()
for _ in range(RATE_LIMIT_CALLS):
    assert _check_rate_limit("test_provider") is True
print(f"  {RATE_LIMIT_CALLS} calls allowed. Passed!")

# ─── Test 3: Remote provider exceeds limit ───
print("\n=== Test 3: Remote Exceeds Limit ===")
result = _check_rate_limit("test_provider")
assert result is False
print(f"  Call {RATE_LIMIT_CALLS + 1} blocked. Passed!")

# ─── Test 4: Different providers have separate limits ───
print("\n=== Test 4: Separate Provider Limits ===")
_remote_call_log.clear()
for _ in range(RATE_LIMIT_CALLS):
    assert _check_rate_limit("provider_a") is True
assert _check_rate_limit("provider_b") is True
print("  Different providers have independent limits. Passed!")

# ─── Test 5: Rate limit constants exist ───
print("\n=== Test 5: Rate Limit Constants ===")
assert RATE_LIMIT_CALLS > 0
assert RATE_LIMIT_WINDOW > 0
print(f"  RATE_LIMIT_CALLS={RATE_LIMIT_CALLS}, RATE_LIMIT_WINDOW={RATE_LIMIT_WINDOW}s. Passed!")

# ─── Cleanup ───
_remote_call_log.clear()

print("\n=== ALL RATE LIMIT TESTS PASSED ===")
