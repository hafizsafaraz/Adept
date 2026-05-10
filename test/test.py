import adept

# Test 1D
a = adept.Tensor([1,2,3,4,5])
assert a.sum() == 15.0
assert a.mean() == 3.0
assert a.max() == 5.0
assert a.min() == 1.0
assert a.shape() == [5]
assert a.ndim() == 1

# Test 2D
b = adept.Tensor([[1,2,3],[4,5,6]])
assert b.shape() == [2, 3]
assert b.ndim() == 2

# Test zeros & ones
z = adept.zeros([2,3])
assert z.get(0) == 0.0
o = adept.ones([2,3])
assert o.get(0) == 1.0

print("All tests passed!")