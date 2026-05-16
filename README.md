<p align="center">
  <img src="branding/logo/logo.svg" width="300"/>
</p>


# Adept

A fast numerical library for Python, written in C++ with Pybind11.

## Version 
0.3.0

## Features

- **Tensor** — supports 1D and 2D tensors
- **Math** — sum, mean, max, min, argmax, argmin, abs, clip, pow, sqrt, stddev, var (raises if tensor is empty, except sum which returns 0.0)
- **Linear Algebra** — dot product for 1D tensors, matmul for 2D tensors
- **Operators** — +, -, *, / between tensors (raises if shape mismatch) and scalars (both `tensor * 2.0` and `2.0 * tensor`)
- **Tensor Operations** — reshape (raises if ndim > 2 or element count mismatch), flatten, transpose (raises if not 2D)
- **Utility** — zeros, ones

## Installation

```bash
git clone https://github.com/hafizsafaraz/adept.git
cd adept
pip install .
```

## Usage

```python
import adept

# Create tensors
a = adept.Tensor([1, 2, 3, 4, 5])
b = adept.Tensor([[1, 2, 3], [4, 5, 6]])

# Math
print(a.sum())     # 15.0
print(a.mean())    # 3.0
print(a.max())     # 5.0
print(a.min())     # 1.0

# Argmax & argmin
c = adept.Tensor([10, 30, 20])
print(c.argmax())  # 1
print(c.argmin())  # 0

# Abs
d = adept.Tensor([-1, 2, -3, 4])
print(d.abs())  # Tensor([1.0, 2.0, 3.0, 4.0])

# Clip
cl = adept.Tensor([0.5, 1, 3, 10, 3])
print(cl.clip(0, 5))  # Tensor([0.5, 1.0, 3.0, 5.0, 3.0])

# Pow
p = adept.Tensor([1, 2, 3, 4, 5])
print(p.pow(2))  # Tensor([1.0, 4.0, 9.0, 16.0, 25.0])

# Sqrt
s = adept.Tensor([1, 4, 9, 16, 25])
print(s.sqrt())  # Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Stddev & Var
sv = adept.Tensor([2, 4, 4, 4, 5, 5, 7, 9])
print(sv.stddev())  # 2.0
print(sv.var())     # 4.0

# Dot product (1D) — returns Tensor
d1 = adept.Tensor([1, 2, 3])
d2 = adept.Tensor([4, 5, 6])
print(d1.dot(d2))  # 32.0

# Matrix multiplication (2D)
m1 = adept.Tensor([[1, 2], [3, 4]])
m2 = adept.Tensor([[5, 6], [7, 8]])
print(m1.matmul(m2))  # Tensor([[19.0, 22.0], [43.0, 50.0]])

# Operators
print(a + a)    # Tensor([2.0, 4.0, 6.0, 8.0, 10.0])
print(a * 2.0)  # Tensor([2.0, 4.0, 6.0, 8.0, 10.0])
print(2.0 * a)  # Tensor([2.0, 4.0, 6.0, 8.0, 10.0])

# Shape info
print(b.shape())   # [2, 3]
print(b.ndim())    # 2

# Tensor operations
e = b.transpose()
f = b.flatten()
g = b.reshape([3, 2])

# Utility
z = adept.zeros([3, 3])
o = adept.ones([3, 3])
```

## Requirements

- Python 3.8+
- C++ compiler (MSVC, GCC, or Clang)
- Pybind11

## License

MIT License — see [LICENSE](LICENSE) for details.