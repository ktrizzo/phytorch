#pragma once

// Forward-mode autodiff via dual numbers.
//
// A Dual carries a value and a fixed-size vector of partial derivatives.
// Models are written as templates on the scalar type, so the same forward()
// code differentiates exactly when instantiated with Dual and runs at full
// double speed when instantiated with double.
//
// Forward mode is the right choice here: physiology fits typically have <20
// parameters but produce vectors of N predictions, and forward mode is O(P)
// per output sample — competitive with reverse mode at this size and far
// simpler to implement.

#include <Eigen/Core>
#include <cmath>
#include <ostream>

namespace phytorch::ad {

template <int N>
struct Dual {
    using Grad = Eigen::Matrix<double, N, 1>;

    double value;
    Grad   grad;

    Dual() : value(0.0), grad(Grad::Zero()) {}
    Dual(double v) : value(v), grad(Grad::Zero()) {}
    Dual(double v, const Grad& g) : value(v), grad(g) {}

    static Dual seed(double v, int i) {
        Grad g = Grad::Zero();
        g(i) = 1.0;
        return Dual(v, g);
    }
};

// ---- arithmetic --------------------------------------------------------

template <int N> Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) { return {a.value + b.value, a.grad + b.grad}; }
template <int N> Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) { return {a.value - b.value, a.grad - b.grad}; }
template <int N> Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) { return {a.value * b.value, a.value * b.grad + b.value * a.grad}; }
template <int N> Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    const double inv = 1.0 / b.value;
    return {a.value * inv, (a.grad - (a.value * inv) * b.grad) * inv};
}
template <int N> Dual<N> operator-(const Dual<N>& a) { return {-a.value, -a.grad}; }

template <int N> Dual<N> operator+(const Dual<N>& a, double b) { return {a.value + b, a.grad}; }
template <int N> Dual<N> operator+(double a, const Dual<N>& b) { return b + a; }
template <int N> Dual<N> operator-(const Dual<N>& a, double b) { return {a.value - b, a.grad}; }
template <int N> Dual<N> operator-(double a, const Dual<N>& b) { return {a - b.value, -b.grad}; }
template <int N> Dual<N> operator*(const Dual<N>& a, double b) { return {a.value * b, a.grad * b}; }
template <int N> Dual<N> operator*(double a, const Dual<N>& b) { return b * a; }
template <int N> Dual<N> operator/(const Dual<N>& a, double b) { return {a.value / b, a.grad / b}; }
template <int N> Dual<N> operator/(double a, const Dual<N>& b) {
    const double inv = 1.0 / b.value;
    return {a * inv, (-a * inv * inv) * b.grad};
}

// ---- transcendentals (overload set used by physiology models) ----------

template <int N> Dual<N> exp(const Dual<N>& a) { const double e = std::exp(a.value); return {e, e * a.grad}; }
template <int N> Dual<N> log(const Dual<N>& a) { return {std::log(a.value), a.grad / a.value}; }
template <int N> Dual<N> sqrt(const Dual<N>& a) { const double s = std::sqrt(a.value); return {s, a.grad / (2.0 * s)}; }
template <int N> Dual<N> pow(const Dual<N>& a, double p) {
    const double v = std::pow(a.value, p - 1.0);
    return {v * a.value, (p * v) * a.grad};
}
template <int N> Dual<N> abs(const Dual<N>& a) {
    return a.value >= 0.0 ? a : -a;
}
template <int N> Dual<N> sin(const Dual<N>& a) { return {std::sin(a.value), std::cos(a.value) * a.grad}; }
template <int N> Dual<N> cos(const Dual<N>& a) { return {std::cos(a.value), -std::sin(a.value) * a.grad}; }

// Digamma ψ(x) = d/dx log Γ(x). Standard recipe: shift to x ≥ 6 via
// ψ(x) = ψ(x+1) - 1/x, then asymptotic series. Accurate to ~1e-12 for
// x > 0, which is the regime any sane physiology fit puts it in.
inline double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    const double x2 = 1.0 / (x * x);
    result += std::log(x) - 0.5 / x
            - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));
    return result;
}

template <int N> Dual<N> lgamma(const Dual<N>& a) {
    return {std::lgamma(a.value), digamma(a.value) * a.grad};
}

// ---- comparisons (compare values only — derivative info is irrelevant) -

template <int N> bool operator<(const Dual<N>& a, const Dual<N>& b) { return a.value <  b.value; }
template <int N> bool operator>(const Dual<N>& a, const Dual<N>& b) { return a.value >  b.value; }
template <int N> bool operator<=(const Dual<N>& a, const Dual<N>& b){ return a.value <= b.value; }
template <int N> bool operator>=(const Dual<N>& a, const Dual<N>& b){ return a.value >= b.value; }
template <int N> bool operator==(const Dual<N>& a, const Dual<N>& b){ return a.value == b.value; }
template <int N> bool operator!=(const Dual<N>& a, const Dual<N>& b){ return a.value != b.value; }
template <int N> bool operator<(const Dual<N>& a, double b)  { return a.value <  b; }
template <int N> bool operator>(const Dual<N>& a, double b)  { return a.value >  b; }
template <int N> bool operator<=(const Dual<N>& a, double b) { return a.value <= b; }
template <int N> bool operator>=(const Dual<N>& a, double b) { return a.value >= b; }
template <int N> bool operator<(double a, const Dual<N>& b)  { return a <  b.value; }
template <int N> bool operator>(double a, const Dual<N>& b)  { return a >  b.value; }
template <int N> bool operator<=(double a, const Dual<N>& b) { return a <= b.value; }
template <int N> bool operator>=(double a, const Dual<N>& b) { return a >= b.value; }

// max/min that preserve gradient flow through whichever branch is selected.
template <int N> Dual<N> max(const Dual<N>& a, const Dual<N>& b) { return a.value >= b.value ? a : b; }
template <int N> Dual<N> min(const Dual<N>& a, const Dual<N>& b) { return a.value <= b.value ? a : b; }
template <int N> Dual<N> max(const Dual<N>& a, double b) { return a.value >= b ? a : Dual<N>(b); }
template <int N> Dual<N> max(double a, const Dual<N>& b) { return max(b, a); }
template <int N> Dual<N> min(const Dual<N>& a, double b) { return a.value <= b ? a : Dual<N>(b); }
template <int N> Dual<N> min(double a, const Dual<N>& b) { return min(b, a); }

template <int N> std::ostream& operator<<(std::ostream& os, const Dual<N>& d) {
    return os << d.value << " (∂=" << d.grad.transpose() << ")";
}

}  // namespace phytorch::ad
