#ifndef VECTOR_H_
#define VECTOR_H_

#include <cmath>

struct Vector3F
{
    float x;
    float y;
    float z;
    constexpr Vector3F() noexcept : x(), y(), z()
    {
    }
    constexpr Vector3F(float x, float y, float z) noexcept : x(x), y(y), z(z)
    {
    }
    constexpr Vector3F operator-() const noexcept
    {
        return Vector3F(-x, -y, -z);
    }
    friend constexpr Vector3F operator+(const Vector3F &a, const Vector3F &b) noexcept
    {
        return Vector3F(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    friend constexpr Vector3F operator-(const Vector3F &a, const Vector3F &b) noexcept
    {
        return Vector3F(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    friend constexpr Vector3F operator*(float factor, const Vector3F &v) noexcept
    {
        return Vector3F(v.x * factor, v.y * factor, v.z * factor);
    }
    friend constexpr Vector3F operator*(const Vector3F &v, float factor) noexcept
    {
        return Vector3F(v.x * factor, v.y * factor, v.z * factor);
    }
    friend constexpr Vector3F operator/(const Vector3F &v, float divisor) noexcept
    {
        return v * (1.0f / divisor);
    }
    constexpr Vector3F &operator+=(const Vector3F &rt) noexcept
    {
        return *this = *this + rt;
    }
    constexpr Vector3F &operator-=(const Vector3F &rt) noexcept
    {
        return *this = *this - rt;
    }
    constexpr Vector3F &operator*=(float rt) noexcept
    {
        return *this = *this * rt;
    }
    constexpr Vector3F &operator/=(float rt) noexcept
    {
        return *this = *this / rt;
    }
    friend constexpr float dot(const Vector3F &a, const Vector3F &b) noexcept
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    friend constexpr Vector3F cross(const Vector3F &a, const Vector3F &b) noexcept
    {
        return Vector3F(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    friend float euclidianNorm(const Vector3F &v) noexcept
    {
        return std::sqrt(dot(v, v));
    }
    friend Vector3F euclidianNormalize(const Vector3F &v) noexcept
    {
        return v / euclidianNorm(v);
    }
};

struct Point3F
{
    float x;
    float y;
    float z;
    constexpr Point3F() noexcept : x(), y(), z()
    {
    }
    constexpr Point3F(float x, float y, float z) noexcept : x(x), y(y), z(z)
    {
    }
    constexpr explicit Point3F(const Vector3F &v) noexcept : x(v.x), y(v.y), z(v.z)
    {
    }
    constexpr explicit operator Vector3F() const noexcept
    {
        return Vector3F(x, y, z);
    }
    friend constexpr Point3F operator+(const Vector3F &v, const Point3F &p) noexcept
    {
        return Point3F(v.x + p.x, v.y + p.y, v.z + p.z);
    }
    friend constexpr Point3F operator+(const Point3F &p, const Vector3F &v) noexcept
    {
        return Point3F(v.x + p.x, v.y + p.y, v.z + p.z);
    }
    friend constexpr Point3F operator-(const Point3F &p, const Vector3F &v) noexcept
    {
        return Point3F(p.x - v.x, p.y - v.y, p.z - v.z);
    }
    friend constexpr Vector3F operator-(const Point3F &a, const Point3F &b) noexcept
    {
        return Vector3F(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    constexpr Point3F &operator+=(const Vector3F &rt) noexcept
    {
        return *this = *this + rt;
    }
    constexpr Point3F &operator-=(const Vector3F &rt) noexcept
    {
        return *this = *this - rt;
    }
};

struct Homogeneous3F
{
    float x;
    float y;
    float z;
    float w;
    constexpr Homogeneous3F() noexcept : x(), y(), z(), w()
    {
    }
    constexpr Homogeneous3F(float x, float y, float z, float w) noexcept : x(x), y(y), z(z), w(w)
    {
    }
    constexpr explicit Homogeneous3F(const Vector3F &v) noexcept : x(v.x), y(v.y), z(v.z), w(0)
    {
    }
    constexpr explicit Homogeneous3F(const Point3F &p) noexcept : x(p.x), y(p.y), z(p.z), w(1)
    {
    }
    constexpr Homogeneous3F operator-() const noexcept
    {
        return Homogeneous3F(-x, -y, -z, -w);
    }
    constexpr Vector3F direction() const noexcept
    {
        return Vector3F(x, y, z);
    }
    constexpr Point3F position() const noexcept
    {
        return static_cast<Point3F>(Vector3F(x, y, z) / w);
    }
    friend constexpr Homogeneous3F operator+(const Homogeneous3F &a,
                                             const Homogeneous3F &b) noexcept
    {
        return Homogeneous3F(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    friend constexpr Homogeneous3F operator-(const Homogeneous3F &a,
                                             const Homogeneous3F &b) noexcept
    {
        return Homogeneous3F(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }
    friend constexpr Homogeneous3F operator*(float factor, const Homogeneous3F &v) noexcept
    {
        return Homogeneous3F(v.x * factor, v.y * factor, v.z * factor, v.w * factor);
    }
    friend constexpr Homogeneous3F operator*(const Homogeneous3F &v, float factor) noexcept
    {
        return Homogeneous3F(v.x * factor, v.y * factor, v.z * factor, v.w * factor);
    }
    friend constexpr Homogeneous3F operator/(const Homogeneous3F &v, float divisor) noexcept
    {
        return v * (1.0f / divisor);
    }
    constexpr Homogeneous3F &operator+=(const Homogeneous3F &rt) noexcept
    {
        return *this = *this + rt;
    }
    constexpr Homogeneous3F &operator-=(const Homogeneous3F &rt) noexcept
    {
        return *this = *this - rt;
    }
    constexpr Homogeneous3F &operator*=(float rt) noexcept
    {
        return *this = *this * rt;
    }
    constexpr Homogeneous3F &operator/=(float rt) noexcept
    {
        return *this = *this / rt;
    }
    friend constexpr float dot(const Homogeneous3F &a, const Homogeneous3F &b) noexcept
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
};

#endif /* VECTOR_H_ */
