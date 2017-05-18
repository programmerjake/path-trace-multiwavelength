#ifndef RAY_H_
#define RAY_H_

#include "vector.h"

struct RayF
{
    Point3F origin;
    Vector3F direction;
    constexpr RayF() noexcept : origin(), direction()
    {
    }
    constexpr RayF(const Point3F &origin, const Vector3F &direction) noexcept : origin(origin),
                                                                                direction(direction)
    {
    }
    constexpr Point3F position(float t) const noexcept
    {
        return origin + t * direction;
    }
};

#endif /* RAY_H_ */
