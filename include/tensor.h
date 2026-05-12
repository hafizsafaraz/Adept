#include <vector>
#include <string>

class Tensor {
    private:
        std::vector<double> data;   // flat storage of all elements
        std::vector<int> shape_;    // size of each dimension
        std::string shape_to_str(std::vector<int> s) const;  // format shape as string for error messages

    public:
        // ── Constructors ──────────────────────────────────
        Tensor(std::vector<double> data_tensor, std::vector<int> shape_tensor); // from flat data + shape
        Tensor(std::vector<double> input);                                       // from 1D list
        Tensor(std::vector<std::vector<double>> input);                          // from 2D list

        // ── Info ──────────────────────────────────────────
        double get(int a) const;              // get element by flat index. throws if index out of range
        std::vector<int> shape() const;       // size of each dimension
        int ndim() const;                     // number of dimensions
        int size() const;                     // total number of elements

        // ── Math ──────────────────────────────────────────
        double sum() const;                   // sum of all elements
        double mean() const;                  // average of all elements. throws if empty
        double max() const;                   // maximum value. throws if empty
        double min() const;                   // minimum value. throws if empty

        // ── Operators (tensor) ────────────────────────────
        Tensor operator+(const Tensor& other) const;  // element-wise add. throws if shape mismatch
        Tensor operator-(const Tensor& other) const;  // element-wise sub. throws if shape mismatch
        Tensor operator*(const Tensor& other) const;  // element-wise mul. throws if shape mismatch
        Tensor operator/(const Tensor& other) const;  // element-wise div. throws if shape mismatch or division by zero

        // ── Operators (scalar) ────────────────────────────
        Tensor operator+(double scalar) const;  // add scalar to every element
        Tensor operator-(double scalar) const;  // subtract scalar from every element
        Tensor operator*(double scalar) const;  // multiply every element by scalar
        Tensor operator/(double scalar) const;  // divide every element by scalar

        // ── Tensor Operations ─────────────────────────────
        Tensor reshape(std::vector<int> new_shape) const;  // reshape to arbitrary new shape. throws if element count mismatch
        Tensor flatten() const;                            // flatten to 1D
        Tensor transpose() const;                          // transpose (2D only). throws if not 2D
};

// ── Utility Functions ─────────────────────────────
Tensor zeros(std::vector<int> shape);   // create tensor filled with zeros
Tensor ones(std::vector<int> shape);    // create tensor filled with ones