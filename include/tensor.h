#include <vector>

class Tensor {
    private:
        std::vector<double> data;   // flat storage of all elements
        std::vector<int> shape_;    // size of each dimension

    public:
        // ── Constructors ──────────────────────────────────
        Tensor(std::vector<double> data_tensor, std::vector<int> shape_tensor); // from flat data + shape
        Tensor(std::vector<double> input);                                       // from 1D list
        Tensor(std::vector<std::vector<double>> input);                          // from 2D list

        // ── Info ──────────────────────────────────────────
        double get(int a) const;              // get element by flat index
        std::vector<int> shape() const;       // size of each dimension
        int ndim() const;                     // number of dimensions
        int size() const;                     // total number of elements

        // ── Math ──────────────────────────────────────────
        double sum() const;                   // sum of all elements
        double mean() const;                  // average of all elements
        double max() const;                   // maximum value
        double min() const;                   // minimum value

        // ── Operators (tensor) ────────────────────────────
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;

        // ── Operators (scalar) ────────────────────────────
        Tensor operator+(double scalar) const;
        Tensor operator-(double scalar) const;
        Tensor operator*(double scalar) const;
        Tensor operator/(double scalar) const;

        // ── Tensor Operations ─────────────────────────────
        Tensor reshape(std::vector<int> new_shape) const;  // reshape to new_rows x new_cols
        Tensor flatten() const;                            // flatten to 1D
        Tensor transpose() const;                          // transpose (2D only)
};

// ── Utility Functions ─────────────────────────────
Tensor zeros(std::vector<int> shape);   // create tensor filled with zeros
Tensor ones(std::vector<int> shape);    // create tensor filled with ones