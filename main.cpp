#include "vector.h"
#include "color.h"
#include "ray.h"
#include "triangle.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <future>
#include <atomic>
#include <mutex>
#include <tuple>
#include <thread>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <chrono>

struct DisplayStatus final
{
    std::uint64_t amountCompleted = -1;
    static constexpr std::uint64_t amountCompletedSteps = 100000;
    std::uint64_t totalSampleCount = 0;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::mutex lock;
    void writeMessage(std::unique_lock<std::mutex> &lockIt)
    {
        auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
        std::cout << amountCompleted * 100.0 / amountCompletedSteps
                  << "%\ttotalSampleCount = " << totalSampleCount << "\t" << elapsedTime.count() << " seconds elapsed" << std::endl;
    }
    void updateAmountCompleted(std::uint64_t newValue)
    {
        std::unique_lock<std::mutex> lockIt(lock);
        amountCompleted = newValue;
        writeMessage(lockIt);
    }
    void updateTotalSampleCount(std::uint64_t newValue)
    {
        std::unique_lock<std::mutex> lockIt(lock);
        totalSampleCount = newValue;
        writeMessage(lockIt);
    }
};

class ImageWriter final
{
private:
    DisplayStatus &displayStatus;
    std::size_t width;
    std::size_t height;
    std::vector<RGBColor> imageData;

public:
    ImageWriter(DisplayStatus &displayStatus, std::size_t width, std::size_t height)
        : displayStatus(displayStatus), width(width), height(height), imageData()
    {
        imageData.resize(width * height);
    }
    template <typename Fn>
    ImageWriter &render(Fn fn)
    {
        auto w = width, h = height; // copy to help optimization
        auto *data = imageData.data();
        std::atomic_size_t pixelsDone(0);
        std::vector<std::future<void>> futures;
        std::size_t totalPixels = w * h;
        displayStatus.updateAmountCompleted(0);
        for(std::size_t y = 0; y < h; y++)
        {
            DisplayStatus &displayStatus = this->displayStatus;
            futures.push_back(std::async(
                std::launch::deferred,
                [y, w, h, data, &pixelsDone, totalPixels, fn, &displayStatus]()
                {
                    for(std::size_t x = 0; x < w; x++)
                    {
                        RGBColor color =
                            fn(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
                        data[x + w * y] = color;
                        std::size_t pixel = pixelsDone.fetch_add(1, std::memory_order_relaxed);
                        std::uint64_t oldAmount = pixel * DisplayStatus::amountCompletedSteps / totalPixels;
                        std::uint64_t newAmount = (pixel + 1) * DisplayStatus::amountCompletedSteps / totalPixels;
                        if(newAmount > oldAmount)
                            displayStatus.updateAmountCompleted(newAmount);
                    }
                }));
        }
        std::atomic_size_t nextFutureIndex(0);
        std::size_t threadCount = std::thread::hardware_concurrency();
        if(threadCount == 0)
            threadCount = 1;
        std::vector<std::thread> threads;
        threads.resize(threadCount);
        for(auto &thread : threads)
        {
            thread = std::thread([&nextFutureIndex, &futures]()
                                 {
                                     while(true)
                                     {
                                         std::size_t futureIndex = nextFutureIndex.fetch_add(
                                             1, std::memory_order_relaxed);
                                         if(futureIndex >= futures.size())
                                             return;
                                         futures[futureIndex].get();
                                     }
                                 });
        }
        for(auto &thread : threads)
        {
            thread.join();
        }
        return *this;
    }
    void writeHDR(std::ostream &os) const
    {
        static_assert(std::is_same<std::uint8_t, unsigned char>::value, "");
        std::vector<std::uint8_t> scanLine(width * 4);
        os << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " << height << " +X " << width << "\n";
        std::uint8_t lineCode[4] = {
            2, 2, static_cast<std::uint8_t>(width >> 8), static_cast<std::uint8_t>(width & 0xFF)};
        for(std::size_t y = 0; y < height; y++)
        {
            os.write(reinterpret_cast<const char *>(lineCode), sizeof(lineCode));
            for(std::size_t x = 0; x < width; x++)
            {
                RGBColor pixel = imageData[x + width * y];
                if(pixel.r < 0)
                    pixel.r = 0;
                if(pixel.g < 0)
                    pixel.g = 0;
                if(pixel.b < 0)
                    pixel.b = 0;
                float maxV = std::fmax(pixel.r, std::fmax(pixel.g, pixel.b)) * (1.0f / 179.0f);
                if(maxV < 1e-30)
                {
                    scanLine[4 * x + 0] = 0;
                    scanLine[4 * x + 1] = 0;
                    scanLine[4 * x + 2] = 0;
                    scanLine[4 * x + 3] = 0;
                    continue;
                }
                int lg = (int)std::ceil(std::log(maxV) / std::log(2.0f) + 1e-5f);
                float scale = std::pow(0.5f, lg - 8) * (1.0f / 179.0f);
                scanLine[4 * x + 0] =
                    std::max(0, std::min(0xFF, static_cast<int>(pixel.r * scale)));
                scanLine[4 * x + 1] =
                    std::max(0, std::min(0xFF, static_cast<int>(pixel.g * scale)));
                scanLine[4 * x + 2] =
                    std::max(0, std::min(0xFF, static_cast<int>(pixel.b * scale)));
                scanLine[4 * x + 3] = lg + 128;
            }
            char buffer[0x80];
            for(std::size_t channel = 0; channel < 4; channel++)
            {
                std::size_t currentRunLength = 0, skipCount = 0;
                for(std::size_t x = 0; x < width;)
                {
                    while(currentRunLength < 0x7F && skipCount <= 0x80
                          && currentRunLength + skipCount + x < width)
                    {
                        while(currentRunLength < 0x7F && currentRunLength + skipCount + x < width
                              && scanLine[channel + 4 * (x + skipCount)]
                                     == scanLine[channel + 4 * (x + skipCount + currentRunLength)])
                            currentRunLength++;
                        if(currentRunLength < 3)
                        {
                            skipCount += currentRunLength;
                            currentRunLength = 0;
                        }
                        else
                            break;
                    }
                    assert(currentRunLength <= 0x7F && currentRunLength + x + skipCount <= width);
                    if(currentRunLength > 0)
                        assert(skipCount <= 0x80);
                    else if(skipCount > 0x80)
                        skipCount = 0x80;
                    if(skipCount > 0)
                    {
                        os.put(static_cast<std::uint8_t>(skipCount));
                        for(std::size_t i = 0; i < skipCount; i++)
                        {
                            buffer[i] = scanLine[channel + 4 * (x + i)];
                        }
                        os.write(buffer, skipCount);
                        x += skipCount;
                        skipCount = 0;
                    }
                    if(currentRunLength > 0)
                    {
                        os.put(static_cast<std::uint8_t>(currentRunLength + 0x80));
                        os.put(scanLine[channel + 4 * (x + skipCount)]);
                        x += currentRunLength;
                        currentRunLength = 0;
                    }
                }
            }
        }
    }
};

struct World
{
    std::default_random_engine re;
    World() noexcept : re()
    {
    }
    float randomFloat() noexcept
    {
        return std::uniform_real_distribution<float>()(re);
    }
    float randomFloat(float maxValue) noexcept
    {
        return std::uniform_real_distribution<float>(0, maxValue)(re);
    }
    float randomFloat(float minValue, float maxValue) noexcept
    {
        return std::uniform_real_distribution<float>(minValue, maxValue)(re);
    }
    Vector3F randomDirection() noexcept
    {
        Vector3F retval;
        while(true)
        {
            retval.x = randomFloat(-1, 1);
            retval.y = randomFloat(-1, 1);
            retval.z = randomFloat(-1, 1);
            float normSquared = dot(retval, retval);
            if(normSquared < 1e-10 || normSquared > 1)
                continue;
            retval /= std::sqrt(normSquared);
            break;
        }
        return retval;
    }
    struct Intersection
    {
        float t;
        std::size_t objectIndex;
        constexpr Intersection() noexcept : t(-1), objectIndex(-1)
        {
        }
        constexpr Intersection(float t, std::size_t objectIndex) noexcept : t(t),
                                                                            objectIndex(objectIndex)
        {
        }
        constexpr int isCloserThan(const Intersection &rt) const noexcept
        {
            return t < 0 ? false : rt.t < 0 ? true : t < rt.t;
        }
    };
    template <typename ChildClass>
    struct GenericDiffuseSurface
    {
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    float intersectionT,
                    World &world,
                    const Vector3F &normal) const noexcept
        {
            auto direction = world.randomDirection();
            auto directionDotNormal = dot(direction, normal);
            if(dot(ray.direction, normal) * directionDotNormal > 0)
            {
                direction = -direction;
                directionDotNormal = -directionDotNormal;
            }
            auto origin = ray.position(intersectionT);
            float intensity =
                static_cast<const ChildClass *>(this)->getIntensity(wavelengthInNanometers);
            return world.trace(RayF(origin, direction),
                               wavelengthInNanometers,
                               intensityMultiplier * intensity * std::fabs(directionDotNormal),
                               depth);
        }
    };
    struct GrayDiffuseSurface : public GenericDiffuseSurface<GrayDiffuseSurface>
    {
        float intensity;
        constexpr explicit GrayDiffuseSurface(float intensity) noexcept : intensity(intensity)
        {
        }
        constexpr float getIntensity(float wavelengthInNanometers) const noexcept
        {
            return intensity;
        }
    };
    struct ColoredDiffuseSurface : public GenericDiffuseSurface<ColoredDiffuseSurface>
    {
        RGBColor color;
        constexpr explicit ColoredDiffuseSurface(const RGBColor &color) noexcept : color(color)
        {
        }
        constexpr float getIntensity(float wavelengthInNanometers) const noexcept
        {
            return color.getIntensityAtWavelength(wavelengthInNanometers);
        }
    };
    struct MirrorSurface
    {
        float intensity;
        constexpr explicit MirrorSurface(float intensity) noexcept : intensity(intensity)
        {
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    float intersectionT,
                    World &world,
                    const Vector3F &normal) const noexcept
        {
            auto direction = ray.direction - 2.0f * dot(normal, ray.direction) * normal;
            auto origin = ray.position(intersectionT);
            return world.trace(RayF(origin, direction),
                               wavelengthInNanometers,
                               intensityMultiplier * intensity,
                               depth);
        }
    };
    struct EmissiveSurface
    {
        float intensity;
        constexpr explicit EmissiveSurface(float intensity) noexcept : intensity(intensity)
        {
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    float intersectionT,
                    World &world,
                    const Vector3F &normal) const noexcept
        {
            return intensityMultiplier * intensity * d65Spectrum.get(wavelengthInNanometers);
        }
    };
    template <typename Surface1, typename Surface2>
    struct MixSurface
    {
        float factor;
        Surface1 surface1;
        Surface2 surface2;
        constexpr MixSurface(float factor, Surface1 surface1, Surface2 surface2) noexcept
            : factor(factor),
              surface1(surface1),
              surface2(surface2)
        {
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    float intersectionT,
                    World &world,
                    const Vector3F &normal) const noexcept
        {
            if(world.randomFloat() < factor)
                return surface1.shade(ray,
                                      wavelengthInNanometers,
                                      intensityMultiplier,
                                      depth,
                                      intersectionT,
                                      world,
                                      normal);
            return surface2.shade(ray,
                                  wavelengthInNanometers,
                                  intensityMultiplier,
                                  depth,
                                  intersectionT,
                                  world,
                                  normal);
        }
    };
    template <typename Surface1, typename Surface2>
    static constexpr MixSurface<Surface1, Surface2> makeMixSurface(float factor,
                                                                   Surface1 surface1,
                                                                   Surface2 surface2) noexcept
    {
        return MixSurface<Surface1, Surface2>(factor, std::move(surface1), std::move(surface2));
    }
    template <typename Object1, typename Object2>
    struct CompositeObject
    {
        static constexpr std::size_t objectIndexCount =
            Object1::objectIndexCount + Object2::objectIndexCount;
        Object1 object1;
        Object2 object2;
        constexpr CompositeObject(Object1 object1, Object2 object2) noexcept : object1(object1),
                                                                               object2(object2)
        {
        }
        Intersection intersect(const RayF &ray) const noexcept
        {
            Intersection object1Intersection = object1.intersect(ray);
            Intersection object2Intersection = object2.intersect(ray);
            if(object1Intersection.isCloserThan(object2Intersection))
                return object1Intersection;
            object2Intersection.objectIndex += Object1::objectIndexCount;
            return object2Intersection;
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    const Intersection &intersection,
                    World &world) const noexcept
        {
            assert(intersection.objectIndex < objectIndexCount);
            if(intersection.objectIndex < Object1::objectIndexCount)
                return object1.shade(
                    ray, wavelengthInNanometers, intensityMultiplier, depth, intersection, world);
            return object2.shade(
                ray,
                wavelengthInNanometers,
                intensityMultiplier,
                depth,
                Intersection(intersection.t, intersection.objectIndex - Object1::objectIndexCount),
                world);
        }
    };
    template <typename Object>
    static constexpr Object makeCompositeObject(Object object) noexcept
    {
        return object;
    }
    template <typename Object1, typename Object2, typename... Objects>
    static constexpr CompositeObject<Object1,
                                     decltype(makeCompositeObject(std::declval<Object2>(),
                                                                  std::declval<Objects>()...))>
        makeCompositeObject(Object1 object1, Object2 object2, Objects... objects) noexcept
    {
        auto rest = makeCompositeObject(std::move(object2), std::move(objects)...);
        return CompositeObject<Object1, decltype(rest)>(std::move(object1), std::move(rest));
    }
    template <typename Surface>
    struct TriangleObject
    {
        static constexpr std::size_t objectIndexCount = 1;
        TriangleF triangle;
        Surface surface;
        constexpr TriangleObject(const TriangleF &triangle, Surface surface) noexcept
            : triangle(triangle),
              surface(std::move(surface))
        {
        }
        Intersection intersect(const RayF &ray) const noexcept
        {
            return Intersection(triangle.intersect(ray), 0);
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    const Intersection &intersection,
                    World &world) const noexcept
        {
            assert(intersection.objectIndex == 0);
            return surface.shade(ray,
                                 wavelengthInNanometers,
                                 intensityMultiplier,
                                 depth,
                                 intersection.t,
                                 world,
                                 euclidianNormalize(triangle.unnormalizedNormal()));
        }
    };
    template <typename Surface>
    static constexpr TriangleObject<Surface> makeTriangleObject(const TriangleF &triangle,
                                                                Surface surface) noexcept
    {
        return TriangleObject<Surface>(triangle, std::move(surface));
    }
    template <typename Surface>
    struct ParallelogramObject
    {
        static constexpr std::size_t objectIndexCount = 1;
        ParallelogramF parallelogram;
        Surface surface;
        constexpr ParallelogramObject(const ParallelogramF &parallelogram, Surface surface) noexcept
            : parallelogram(parallelogram),
              surface(std::move(surface))
        {
        }
        Intersection intersect(const RayF &ray) const noexcept
        {
            return Intersection(parallelogram.intersect(ray), 0);
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    const Intersection &intersection,
                    World &world) const noexcept
        {
            assert(intersection.objectIndex == 0);
            return surface.shade(ray,
                                 wavelengthInNanometers,
                                 intensityMultiplier,
                                 depth,
                                 intersection.t,
                                 world,
                                 euclidianNormalize(parallelogram.unnormalizedNormal()));
        }
    };
    template <typename Surface>
    static constexpr ParallelogramObject<Surface> makeParallelogramObject(
        const ParallelogramF &parallelogram, Surface surface) noexcept
    {
        return ParallelogramObject<Surface>(parallelogram, std::move(surface));
    }
    static constexpr auto makeCornellBoxObject()
    {
        constexpr auto leftFace =
            makeParallelogramObject(ParallelogramF({-1, -1, -2}, {-1, 1, -2}, {-1, -1, 0}),
                                    ColoredDiffuseSurface({1, 0, 0}));
        constexpr auto rightFace = makeParallelogramObject(
            ParallelogramF({1, -1, -2}, {1, -1, 0}, {1, 1, -2}), ColoredDiffuseSurface({0, 1, 0}));
        constexpr auto topFace = makeParallelogramObject(
            ParallelogramF({-1, 1, -2}, {1, 1, -2}, {-1, 1, 0}), GrayDiffuseSurface(1));
        constexpr auto bottomFace = makeParallelogramObject(
            ParallelogramF({-1, -1, -2}, {-1, -1, 0}, {1, -1, -2}), GrayDiffuseSurface(1));
        constexpr auto backFace = makeParallelogramObject(
            ParallelogramF({-1, -1, -2}, {1, -1, -2}, {-1, 1, -2}), GrayDiffuseSurface(1));
        constexpr float lightSize = 0.5f;
        constexpr float lightY = 1 - 1.0 / 1024;
        constexpr auto light =
            makeParallelogramObject(ParallelogramF({-lightSize, lightY, -1 - lightSize},
                                                   {lightSize, lightY, -1 - lightSize},
                                                   {-lightSize, lightY, -1 + lightSize}),
                                    EmissiveSurface(10));
        return makeCompositeObject(leftFace, rightFace, bottomFace, topFace, backFace, light);
    }
    struct RayDirectionBackgroundObject
    {
        static constexpr std::size_t objectIndexCount = 1;
        Intersection intersect(const RayF &ray) const noexcept
        {
            return Intersection(1e30, 0);
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    const Intersection &intersection,
                    World &world) const noexcept
        {
            assert(intersection.objectIndex == 0);
            auto v = euclidianNormalize(ray.direction);
            v = v * -0.5f + Vector3F(0.5f, 0.5f, 0.5f);
            return intensityMultiplier
                   * RGBColor(v.x, v.y, v.z).getIntensityAtWavelength(wavelengthInNanometers);
        }
    };
    struct BlackBackgroundObject
    {
        static constexpr std::size_t objectIndexCount = 1;
        Intersection intersect(const RayF &ray) const noexcept
        {
            return Intersection(1e30, 0);
        }
        float shade(const RayF &ray,
                    float wavelengthInNanometers,
                    float intensityMultiplier,
                    std::size_t depth,
                    const Intersection &intersection,
                    World &world) const noexcept
        {
            assert(intersection.objectIndex == 0);
            return 0;
        }
    };
    template <typename Object>
    float traceHelper(const RayF &ray,
                      float wavelengthInNanometers,
                      float intensityMultiplier,
                      std::size_t depth,
                      const Object &object) noexcept
    {
        if(depth++ > 10)
            return 0;
        return object.shade(
            ray, wavelengthInNanometers, intensityMultiplier, depth, object.intersect(ray), *this);
    }
    float trace(const RayF &ray,
                float wavelengthInNanometers,
                float intensityMultiplier,
                std::size_t depth = 0) noexcept
    {
        constexpr auto object =
            makeCompositeObject(makeCornellBoxObject(), BlackBackgroundObject());
        return traceHelper(ray, wavelengthInNanometers, intensityMultiplier, depth, object);
    }
};

int main()
{
    constexpr std::size_t width = 1920;
    constexpr std::size_t height = 1080;
    constexpr float targetAccuracy = 0.5 / 0x100;
    constexpr float xScale = width > height ? static_cast<float>(width) / height : 1.0f;
    constexpr float yScale = width > height ? 1.0f : static_cast<float>(height) / width;
    constexpr Point3F origin(0, 0, 0);
    constexpr std::size_t maxSampleCount = 1ULL << 22;
    constexpr std::uint64_t displaySampleCountPeriod = 0x100000;
    DisplayStatus displayStatus;
    ImageWriter imageWriter(displayStatus, width, height);
    std::atomic_uint_fast64_t totalSampleCount(0);
    imageWriter.render(
        [&totalSampleCount, &displayStatus](std::size_t ix, std::size_t iy) -> RGBColor
        {
            static thread_local std::vector<RGBColor> sampleColors(maxSampleCount);
            World world;
            world.re.seed(ix + iy * 0x621347UL);
            std::size_t sampleCount = maxSampleCount >> 10;
            RGBColor sumColor{};
            RGBColor averageColor;
            std::size_t sample = 0;
            while(true)
            {
                for(; sample < sampleCount; sample++)
                {
                    float fxu = ix + world.randomFloat();
                    float fyu = iy + world.randomFloat();
                    fxu /= width;
                    fyu /= height;
                    float fx = (fxu * 2.0f - 1.0f) * xScale;
                    float fy = (1.0f - fyu * 2.0f) * yScale;
                    RayF ray(origin, Vector3F(fx, fy, -1));
                    float wavelengthInNanometers =
                        world.randomFloat(CIEColorMatchingFunction::minWavelengthInNanometers,
                                          CIEColorMatchingFunction::maxWavelengthInNanometers);
                    auto color = static_cast<RGBColor>(WavelengthIntensityColor(
                        wavelengthInNanometers, world.trace(ray, wavelengthInNanometers, 1.0f)));
                    sampleColors[sample] = color;
                    sumColor += color;
                }
                float invSampleCount = 1.0f / sampleCount;
                averageColor = sumColor * invSampleCount;
                if(sampleCount >= maxSampleCount)
                    break;
                float totalDeviation = 0;
                for(std::size_t i = 0; i < sampleCount; i++)
                    totalDeviation += std::fabs(sampleColors[i].r - averageColor.r)
                                      + std::fabs(sampleColors[i].g - averageColor.g)
                                      + std::fabs(sampleColors[i].b - averageColor.b);
                float standardDeviationOfMean =
                    totalDeviation * invSampleCount / std::sqrt(static_cast<float>(sampleCount));
                if(standardDeviationOfMean < targetAccuracy)
                    break;
                sampleCount *= 2;
                if(sampleCount > maxSampleCount)
                    sampleCount = maxSampleCount;
            }
            std::uint64_t lastSampleCount =
                totalSampleCount.fetch_add(sampleCount, std::memory_order_relaxed);
            std::uint64_t newSampleCount = lastSampleCount + sampleCount;
            if(lastSampleCount / displaySampleCountPeriod < newSampleCount
                                                                / displaySampleCountPeriod)
            {
                displayStatus.updateTotalSampleCount(newSampleCount);
            }
            auto outputColor = sumColor * (1.0f / sampleCount);
#if 0
                    outputColor.b = static_cast<float>(sampleCount) / maxSampleCount;
#endif
            return outputColor;
        });
    displayStatus.updateTotalSampleCount(totalSampleCount.load(std::memory_order_relaxed));
    std::ofstream os("out.hdr", std::ios::binary);
    imageWriter.writeHDR(os);
}
