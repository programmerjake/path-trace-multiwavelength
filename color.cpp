#include "color.h"
#include <cstdlib>
#include <iostream>

constexpr CIEColorMatchingFunction XYZColor::cieXColorMatchingFunction;
constexpr CIEColorMatchingFunction XYZColor::cieYColorMatchingFunction;
constexpr CIEColorMatchingFunction XYZColor::cieZColorMatchingFunction;

constexpr CIEColorMatchingFunction RGBColor::srgbRColorMatchingFunction;
constexpr CIEColorMatchingFunction RGBColor::srgbGColorMatchingFunction;
constexpr CIEColorMatchingFunction RGBColor::srgbBColorMatchingFunction;

constexpr float SRGBColorF::linearToGammaTransitionPoint;
constexpr float SRGBColorF::linearToGammaLinearFactor;
constexpr float SRGBColorF::linearToGammaA;
constexpr float SRGBColorF::linearToGammaExponent;
constexpr float SRGBColorF::linearToGammaPowerFactor;
constexpr float SRGBColorF::gammaToLinearLinearFactor;
constexpr float SRGBColorF::gammaToLinearTransitionPoint;
constexpr float SRGBColorF::gammaToLinearA;
constexpr float SRGBColorF::gammaToLinearExponent;
constexpr float SRGBColorF::gammaToLinearBaseFactor;

#if 0
namespace
{
void writeRGBColor(std::ostream &os, RGBColor color)
{
    os << "<" << color.r << "," << color.g << "," << color.b << ">";
}

constexpr std::size_t supersampleCount = 1024;

template <typename SpectrumFn>
RGBColor getRGBColorFromSpectrum(SpectrumFn spectrumFn)
{
    RGBColor result{};
    constexpr std::size_t stepCount = CIEColorMatchingFunction::valueCount * supersampleCount;
    constexpr float stepSize = 1.0 / supersampleCount;
    float wavelength = CIEColorMatchingFunction::minWavelengthInNanometers;
    for(std::size_t i = 0; i < stepCount; i++, wavelength += stepSize)
    {
        float intensity = spectrumFn(wavelength);
        result += intensity * RGBColor::fromWavelength(wavelength);
    }
    result *= 1.0f / stepCount;
    return result;
}

int test()
{
    constexpr RGBColor testColors[] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    };
    std::cout.precision(8);
    for(auto testColor : testColors)
    {
        writeRGBColor(std::cout, testColor);
        std::cout << " -> ";
        writeRGBColor(
            std::cout,
            getRGBColorFromSpectrum([&](float wavelength)
                                    {
                                        return testColor.getIntensityAtWavelength(wavelength);
                                    }));
        std::cout << std::endl;
    }
    std::cout << "d65 -> ";
    writeRGBColor(
        std::cout,
        getRGBColorFromSpectrum([](float wavelength)
                                {
                                    return d65Spectrum.get(wavelength);
                                }));
    std::cout << std::endl;
    return 0;
}

struct Init
{
    Init()
    {
        std::exit(test());
    }
} init;
}
#endif
