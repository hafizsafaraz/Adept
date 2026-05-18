import acumen

# ── 1D ────────────────────────────────────────────
a = acumen.Tensor([1, 2, 3, 4, 5])
assert a.sum() == 15.0
assert a.mean() == 3.0
assert a.max() == 5.0
assert a.min() == 1.0
assert a.shape() == [5]
assert a.ndim() == 1

# ── sum(axis) ─────────────────────────────────────
sa = acumen.Tensor([[1, 2, 3], [4, 5, 6]])
assert sa.sum(1).get(0) == 6.0   # first row: 1+2+3
assert sa.sum(1).get(1) == 15.0  # second row: 4+5+6
assert sa.sum(0).get(0) == 5.0   # first col: 1+4
assert sa.sum(0).get(1) == 7.0   # second col: 2+5
assert sa.sum(0).get(2) == 9.0   # third col: 3+6

try:
    sa.sum(2)
    assert False, "Should have raised"
except Exception:
    pass

# ── mean(axis) ────────────────────────────────────
ma = acumen.Tensor([[1, 2, 3], [4, 5, 6]])
assert ma.mean(1).get(0) == 2.0  # first row: (1+2+3)/3
assert ma.mean(1).get(1) == 5.0  # second row: (4+5+6)/3
assert ma.mean(0).get(0) == 2.5  # first col: (1+4)/2
assert ma.mean(0).get(1) == 3.5  # second col: (2+5)/2
assert ma.mean(0).get(2) == 4.5  # third col: (3+6)/2

try:
    acumen.Tensor([1, 2, 3]).mean(0)
    assert False, "Should have raised"
except Exception:
    pass

try:
    ma.mean(2)
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([]).mean(0)
    assert False, "Should have raised"
except Exception:
    pass



# ── 2D ────────────────────────────────────────────
b = acumen.Tensor([[1, 2, 3], [4, 5, 6]])
assert b.shape() == [2, 3]
assert b.ndim() == 2

# ── zeros & ones ──────────────────────────────────
z = acumen.zeros([2, 3])
assert z.get(0) == 0.0
o = acumen.ones([2, 3])
assert o.get(0) == 1.0

# ── get() ─────────────────────────────────────────
assert a.get(0) == 1.0
assert a.get(4) == 5.0

# ── Tensor operators ──────────────────────────────
x = acumen.Tensor([1, 2, 3])
y = acumen.Tensor([4, 5, 6])
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
r = acumen.Tensor([1, 2, 3, 4, 5, 6])
assert r.reshape([2, 3]).shape() == [2, 3]
assert r.reshape([3, 2]).shape() == [3, 2]
assert r.flatten().shape() == [6]

# ── transpose ─────────────────────────────────────
t = acumen.Tensor([[1, 2, 3], [4, 5, 6]])
t2 = t.transpose()
assert t2.shape() == [3, 2]
assert t2.get(0) == 1.0
assert t2.get(1) == 4.0

# ── dot ───────────────────────────────────────────

# 1D
d = acumen.Tensor([1, 2, 3])
d2 = acumen.Tensor([4, 5, 6])
assert d.dot(d2).get(0) == 32.0

# 2D
m1 = acumen.Tensor([[1, 2], [3, 4]])
m2 = acumen.Tensor([[5, 6], [7, 8]])
dm = m1.dot(m2)
assert dm.shape() == [2, 2]
assert dm.get(0) == 19.0
assert dm.get(1) == 22.0
assert dm.get(2) == 43.0
assert dm.get(3) == 50.0

# shape mismatch 1D
try:
    acumen.Tensor([1, 2]).dot(acumen.Tensor([1, 2, 3]))
    assert False, "Should have raised"
except Exception:
    pass

# shape mismatch 2D
try:
    acumen.Tensor([[1, 2], [3, 4]]).dot(acumen.Tensor([[1, 2], [3, 4], [5, 6]]))
    assert False, "Should have raised"
except Exception:
    pass

# ── argmax & argmin ───────────────────────────────
a2 = acumen.Tensor([10, 30, 20])
assert a2.argmax() == 1
assert a2.argmin() == 0

# empty tensor
try:
    acumen.Tensor([]).argmax()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([]).argmin()
    assert False, "Should have raised"
except Exception:
    pass

# ── abs ───────────────────────────────────────────
ab = acumen.Tensor([-1, 2, -3, 4])
assert ab.abs().get(0) == 1.0
assert ab.abs().get(1) == 2.0
assert ab.abs().get(2) == 3.0

try:
    acumen.Tensor([]).abs()
    assert False, "Should have raised"
except Exception:
    pass

# ── clip ──────────────────────────────────────────
cl = acumen.Tensor([0.5, 1, 3, 10, 3])
assert cl.clip(0, 5).get(3) == 5.0
assert cl.clip(0, 5).get(0) == 0.5

try:
    acumen.Tensor([]).clip(0, 5)
    assert False, "Should have raised"
except Exception:
    pass

try:
    cl.clip(5, 0)
    assert False, "Should have raised"
except Exception:
    pass

# ── pow ───────────────────────────────────────────
pw = acumen.Tensor([1, 2, 3, 4, 5])
assert pw.pow(2).get(0) == 1.0
assert pw.pow(2).get(1) == 4.0
assert pw.pow(2).get(2) == 9.0

try:
    acumen.Tensor([]).pow(2)
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([0, 1, 2]).pow(-1)
    assert False, "Should have raised"
except Exception:
    pass

# ── sqrt ──────────────────────────────────────────
sq = acumen.Tensor([1, 4, 9, 16, 25])
assert sq.sqrt().get(0) == 1.0
assert sq.sqrt().get(1) == 2.0
assert sq.sqrt().get(2) == 3.0

try:
    acumen.Tensor([]).sqrt()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([-1, 2, 3]).sqrt()
    assert False, "Should have raised"
except Exception:
    pass

# ── stddev & var ──────────────────────────────────
sv = acumen.Tensor([2, 4, 4, 4, 5, 5, 7, 9])
assert sv.stddev() == 2.0
assert sv.var() == 4.0

try:
    acumen.Tensor([]).stddev()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([]).var()
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
    x / acumen.Tensor([0, 0, 0])
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
    x + acumen.Tensor([1, 2])
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
        method(acumen.Tensor([]))
        assert False, "Should have raised"
    except Exception:
        pass

# jagged array
try:
    acumen.Tensor([[1, 2], [3, 4, 5]])
    assert False, "Should have raised"
except Exception:
    pass

# ── normalize ─────────────────────────────────────
nm = acumen.Tensor([1, 2, 3, 4, 5])
assert nm.normalize().get(0) == 0.0
assert nm.normalize().get(4) == 1.0

try:
    acumen.Tensor([]).normalize()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([5, 5, 5]).normalize()
    assert False, "Should have raised"
except Exception:
    pass

# ── log ───────────────────────────────────────────
lg = acumen.Tensor([1, 2, 3])
assert lg.log().get(0) == 0.0

try:
    acumen.Tensor([]).log()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([0, 1, 2]).log()
    assert False, "Should have raised"
except Exception:
    pass

try:
    acumen.Tensor([-1, 1, 2]).log()
    assert False, "Should have raised"
except Exception:
    pass

print("All tests passed!")