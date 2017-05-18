#include "color.h"

constexpr int CIEColorMatchingFunction::minWavelengthInNanometers;
constexpr int CIEColorMatchingFunction::maxWavelengthInNanometers;
constexpr std::size_t CIEColorMatchingFunction::valueCount;

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
