import adept

# ── 1D ────────────────────────────────────────────
a = adept.Tensor([1, 2, 3, 4, 5])
assert a.sum() == 15.0
assert a.mean() == 3.0
assert a.max() == 5.0
assert a.min() == 1.0
assert a.shape() == [5]
assert a.ndim() == 1

# ── 2D ────────────────────────────────────────────
b = adept.Tensor([[1, 2, 3], [4, 5, 6]])
assert b.shape() == [2, 3]
assert b.ndim() == 2

# ── zeros & ones ──────────────────────────────────
z = adept.zeros([2, 3])
assert z.get(0) == 0.0
o = adept.ones([2, 3])
assert o.get(0) == 1.0

# ── get() ─────────────────────────────────────────
assert a.get(0) == 1.0
assert a.get(4) == 5.0

# ── Tensor operators ──────────────────────────────
x = adept.Tensor([1, 2, 3])
y = adept.Tensor([4, 5, 6])
assert (x + y).get(0) == 5.0
assert (x - y).get(0) == -3.0
assert (x * y).get(0) == 4.0
assert (x / y).get(0) == 0.25

# ── Scalar operators ──────────────────────────────
assert (x + 1.0).get(0) == 2.0
assert (x - 1.0).get(0) == 0.0
assert (x * 2.0).get(0) == 2.0
assert (x / 2.0).get(0) == 0.5

# ── reshape & flatten ─────────────────────────────
r = adept.Tensor([1, 2, 3, 4, 5, 6])
assert r.reshape([2, 3]).shape() == [2, 3]
assert r.reshape([3, 2]).shape() == [3, 2]
assert r.flatten().shape() == [6]

# ── transpose ─────────────────────────────────────
t = adept.Tensor([[1, 2, 3], [4, 5, 6]])
t2 = t.transpose()
assert t2.shape() == [3, 2]
assert t2.get(0) == 1.0
assert t2.get(1) == 4.0

# ── Edge cases ────────────────────────────────────

# get() out of range
try:
    a.get(99)
    assert False, "Should have raised"
except Exception:
    pass

# division by zero (tensor)
try:
    x / adept.Tensor([0, 0, 0])
    assert False, "Should have raised"
except Exception:
    pass

# division by zero (scalar)
try:
    x / 0.0
    assert False, "Should have raised"
except Exception:
    pass

# shape mismatch
try:
    x + adept.Tensor([1, 2])
    assert False, "Should have raised"
except Exception:
    pass

# reshape mismatch
try:
    r.reshape([2, 2])
    assert False, "Should have raised"
except Exception:
    pass

# transpose non-2D
try:
    a.transpose()
    assert False, "Should have raised"
except Exception:
    pass

# empty tensor
try:
    adept.Tensor([]).max()
    assert False, "Should have raised"
except Exception:
    pass

# jagged array
try:
    adept.Tensor([[1, 2], [3, 4, 5]])
    assert False, "Should have raised"
except Exception:
    pass

print("All tests passed!")