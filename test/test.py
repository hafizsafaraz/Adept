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

# ── dot ───────────────────────────────────────────

# 1D
d = adept.Tensor([1, 2, 3])
d2 = adept.Tensor([4, 5, 6])
assert d.dot(d2).get(0) == 32.0

# 2D
m1 = adept.Tensor([[1, 2], [3, 4]])
m2 = adept.Tensor([[5, 6], [7, 8]])
dm = m1.dot(m2)
assert dm.shape() == [2, 2]
assert dm.get(0) == 19.0
assert dm.get(1) == 22.0
assert dm.get(2) == 43.0
assert dm.get(3) == 50.0

# shape mismatch 1D
try:
    adept.Tensor([1, 2]).dot(adept.Tensor([1, 2, 3]))
    assert False, "Should have raised"
except Exception:
    pass

# shape mismatch 2D
try:
    adept.Tensor([[1, 2], [3, 4]]).dot(adept.Tensor([[1, 2], [3, 4], [5, 6]]))
    assert False, "Should have raised"
except Exception:
    pass

# ── argmax & argmin ───────────────────────────────
a2 = adept.Tensor([10, 30, 20])
assert a2.argmax() == 1
assert a2.argmin() == 0

# empty tensor
try:
    adept.Tensor([]).argmax()
    assert False, "Should have raised"
except Exception:
    pass

try:
    adept.Tensor([]).argmin()
    assert False, "Should have raised"
except Exception:
    pass

# ── abs ───────────────────────────────────────────
ab = adept.Tensor([-1, 2, -3, 4])
assert ab.abs().get(0) == 1.0
assert ab.abs().get(1) == 2.0
assert ab.abs().get(2) == 3.0

try:
    adept.Tensor([]).abs()
    assert False, "Should have raised"
except Exception:
    pass

# ── clip ──────────────────────────────────────────
cl = adept.Tensor([0.5, 1, 3, 10, 3])
assert cl.clip(0, 5).get(3) == 5.0
assert cl.clip(0, 5).get(0) == 0.5

try:
    adept.Tensor([]).clip(0, 5)
    assert False, "Should have raised"
except Exception:
    pass

try:
    cl.clip(5, 0)
    assert False, "Should have raised"
except Exception:
    pass

# ── pow ───────────────────────────────────────────
pw = adept.Tensor([1, 2, 3, 4, 5])
assert pw.pow(2).get(0) == 1.0
assert pw.pow(2).get(1) == 4.0
assert pw.pow(2).get(2) == 9.0

try:
    adept.Tensor([]).pow(2)
    assert False, "Should have raised"
except Exception:
    pass

try:
    adept.Tensor([0, 1, 2]).pow(-1)
    assert False, "Should have raised"
except Exception:
    pass

# ── sqrt ──────────────────────────────────────────
sq = adept.Tensor([1, 4, 9, 16, 25])
assert sq.sqrt().get(0) == 1.0
assert sq.sqrt().get(1) == 2.0
assert sq.sqrt().get(2) == 3.0

try:
    adept.Tensor([]).sqrt()
    assert False, "Should have raised"
except Exception:
    pass

try:
    adept.Tensor([-1, 2, 3]).sqrt()
    assert False, "Should have raised"
except Exception:
    pass

# ── stddev & var ──────────────────────────────────
sv = adept.Tensor([2, 4, 4, 4, 5, 5, 7, 9])
assert sv.stddev() == 2.0
assert sv.var() == 4.0

try:
    adept.Tensor([]).stddev()
    assert False, "Should have raised"
except Exception:
    pass

try:
    adept.Tensor([]).var()
    assert False, "Should have raised"
except Exception:
    pass

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
for method in [lambda t: t.sum(), lambda t: t.mean(), lambda t: t.max(), lambda t: t.min()]:
    try:
        method(adept.Tensor([]))
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