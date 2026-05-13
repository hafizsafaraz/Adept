# Adept

A fast numerical library for Python, written in C++ with Pybind11.

## Version 
0.1.5

## Features

- **Tensor** — supports 1D and 2D tensors
- **Math** — sum, mean, max, min (raises if tensor is empty)
- **Operators** — +, -, *, / between tensors (raises if shape mismatch) and scalars
- **Tensor Operations** — reshape (raises if element count mismatch), flatten, transpose (raises if not 2D)
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

# Operators
c = a + a          # [2, 4, 6, 8, 10]
d = a * 2.0        # [2, 4, 6, 8, 10]

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