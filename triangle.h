#ifndef TRIANGLE_H_
#define TRIANGLE_H_

#include "vector.h"
#include "ray.h"

template <bool IsParallelogram>
struct TriangleOrParallelogramF
{
    Point3F p1;
    Point3F p2;
    Point3F p3;
    constexpr TriangleOrParallelogramF() noexcept : p1(), p2(), p3()
    {
    }
    constexpr TriangleOrParallelogramF(const Point3F &p1, const Point3F &p2, const Point3F &p3) noexcept : p1(p1),
                                                                                            p2(p2),
                                                                                            p3(p3)
    {
    }
    constexpr Vector3F unnormalizedNormal() const noexcept
    {
        return cross(p2 - p1, p3 - p1);
    }
    constexpr TriangleOrParallelogramF reversed() const noexcept
    {
        return TriangleOrParallelogramF(p1, p3, p2);
    }
    constexpr float intersect(const RayF &ray,
                              float defaultReturnValue = -1,
                              bool doubleSided = false,
                              float epsilon = 1e-5) const noexcept
    {
        auto p = cross(ray.direction, p3 - p1);
        float determinate = dot(p2 - p1, p);
        // determinate is composed of triple products, adjust epsilon too
        epsilon = epsilon * epsilon * epsilon;
        if((!doubleSided || determinate > -epsilon) && determinate < epsilon)
            return defaultReturnValue;
        float inverseDeterminate = 1 / determinate;
        auto offset = ray.origin - p1;
        float u = dot(offset, p) * inverseDeterminate;
        if(u < 0 || !(u < 1))
            return defaultReturnValue;
        auto qVector = cross(offset, p2 - p1);
        float v = dot(ray.direction, qVector) * inverseDeterminate;
        if(IsParallelogram)
        {
            if(v < 0 || !(v < 1))
                return defaultReturnValue;
        }
        else
        {
            if(v < 0 || !(u + v < 1))
                return defaultReturnValue;
        }
        return dot(p3 - p1, qVector) * inverseDeterminate;
    }
};

typedef TriangleOrParallelogramF<false> TriangleF;
typedef TriangleOrParallelogramF<true> ParallelogramF;

#endif /* TRIANGLE_H_ */
