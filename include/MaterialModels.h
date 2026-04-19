#pragma once
/**
 * @file MaterialModels.h
 * @brief Material models for cutting simulation
 * 
 * - Johnson-Cook plasticity model
 * - Usui wear model
 * - Thermal coupling
 */

#include "Types.h"

namespace edgepredict {

/**
 * @brief Johnson-Cook plasticity parameters
 */
struct JohnsonCookParams {
    double A;       // Yield strength (Pa)
    double B;       // Strain hardening coefficient (Pa)
    double n;       // Strain hardening exponent
    double C;       // Strain rate sensitivity
    double m;       // Thermal softening exponent
    double T_melt;  // Melting temperature (°C)
    double T_ref;   // Reference temperature (°C)
    double strainRate_ref; // Reference strain rate (1/s)
    
    EP_HOST_DEVICE JohnsonCookParams()
        : A(880e6), B(290e6), n(0.47), C(0.015), m(1.0),
          T_melt(1660.0), T_ref(25.0), strainRate_ref(1.0) {}
};

/**
 * @brief Usui wear model parameters
 */
struct UsuiWearParams {
    double A;       // Wear coefficient
    double B;       // Activation energy parameter
    
    EP_HOST_DEVICE UsuiWearParams() : A(1e-9), B(1000.0) {}
};

/**
 * @brief Failure criterion parameters
 */
struct FailureParams {
    double D1, D2, D3, D4, D5;  // JC damage parameters
    double shearModulus;
    double fractureToughness;
    
    EP_HOST_DEVICE FailureParams()
        : D1(0.05), D2(3.44), D3(-2.12), D4(0.002), D5(0.61),
          shearModulus(42e9), fractureToughness(55e6) {}
};

// ============================================================================
// Johnson-Cook Flow Stress Model
// ============================================================================

/**
 * @brief Compute Johnson-Cook flow stress
 * 
 * σ = (A + B·ε^n) · (1 + C·ln(ε̇*)) · (1 - T*^m)
 * 
 * @param strain Equivalent plastic strain
 * @param strainRate Strain rate (1/s)
 * @param temperature Current temperature (°C)
 * @param params JC parameters
 * @return Flow stress (Pa)
 */
EP_HOST_DEVICE inline double calculateJohnsonCookStress(
    double strain, double strainRate, double temperature,
    const JohnsonCookParams& params
) {
    // Strain hardening: (A + B·ε^n)
    double strainTerm = params.A + params.B * pow(strain, params.n);
    
    // Strain rate sensitivity: (1 + C·ln(ε̇*))
    double strainRateRatio = strainRate / params.strainRate_ref;
    if (strainRateRatio < 1.0) strainRateRatio = 1.0;
    double strainRateTerm = 1.0 + params.C * log(strainRateRatio);
    
    // Thermal softening: (1 - T*^m)
    double T_star = 0.0;
    if (temperature > params.T_ref) {
        T_star = (temperature - params.T_ref) / (params.T_melt - params.T_ref);
        if (T_star > 1.0) T_star = 1.0;
    }
    double thermalTerm = 1.0 - pow(T_star, params.m);
    if (thermalTerm < 0.01) thermalTerm = 0.01;  // Prevent zero stress
    
    return strainTerm * strainRateTerm * thermalTerm;
}

// ============================================================================
// Usui Wear Model
// ============================================================================

/**
 * @brief Calculate Usui wear rate
 * 
 * dw/dt = A · σ · v · exp(-B/T)
 * 
 * @param stress Normal stress at interface (Pa)
 * @param velocity Sliding velocity (m/s)
 * @param temperature Interface temperature (°C)
 * @param params Usui parameters
 * @return Wear rate (m/s)
 */
EP_HOST_DEVICE inline double calculateUsuiWearRate(
    double stress, double velocity, double temperature,
    const UsuiWearParams& params
) {
    if (stress <= 0 || velocity <= 0) return 0.0;
    
    // Temperature in Kelvin
    double T_K = temperature + 273.15;
    if (T_K < 300.0) T_K = 300.0;
    
    // Arrhenius-type temperature dependence
    double tempFactor = exp(-params.B / T_K);
    
    return params.A * stress * velocity * tempFactor;
}

// ============================================================================
// Damage and Failure
// ============================================================================

/**
 * @brief Calculate damage increment (JC damage model)
 * 
 * D = Σ(Δε / εf)  where  εf = [D1 + D2·exp(D3·σ*)]·[1 + D4·ln(ε̇*)]·[1 + D5·T*]
 * 
 * @param strainIncrement Plastic strain increment
 * @param stressTriaxiality Stress triaxiality (σm/σeq)
 * @param strainRate Strain rate (1/s)
 * @param temperature Temperature (°C)
 * @param params Failure parameters
 * @param jcParams JC parameters (for T_melt, T_ref)
 * @return Damage increment
 */
EP_HOST_DEVICE inline double calculateDamageIncrement(
    double strainIncrement,
    double stressTriaxiality,
    double strainRate,
    double temperature,
    const FailureParams& params,
    const JohnsonCookParams& jcParams
) {
    // Triaxiality dependence
    double triaxTerm = params.D1 + params.D2 * exp(params.D3 * stressTriaxiality);
    if (triaxTerm < 0.01) triaxTerm = 0.01;
    
    // Strain rate dependence
    double strainRateRatio = strainRate / jcParams.strainRate_ref;
    if (strainRateRatio < 1.0) strainRateRatio = 1.0;
    double rateTerm = 1.0 + params.D4 * log(strainRateRatio);
    
    // Temperature dependence
    double T_star = 0.0;
    if (temperature > jcParams.T_ref) {
        T_star = (temperature - jcParams.T_ref) / (jcParams.T_melt - jcParams.T_ref);
        if (T_star > 1.0) T_star = 1.0;
    }
    double tempTerm = 1.0 + params.D5 * T_star;
    
    // Fracture strain
    double fracStrain = triaxTerm * rateTerm * tempTerm;
    if (fracStrain < 0.001) fracStrain = 0.001;
    
    // Damage increment
    return strainIncrement / fracStrain;
}

// ============================================================================
// Thermal Calculations
// ============================================================================

/**
 * @brief Calculate heat generation from plastic work
 * 
 * Q = β · σ · dε / dt
 * 
 * @param stress Flow stress (Pa)
 * @param strainRate Strain rate (1/s)
 * @param taylorQuinney Taylor-Quinney coefficient (typically 0.9)
 * @return Heat generation rate (W/m³)
 */
EP_HOST_DEVICE inline double calculatePlasticHeatGeneration(
    double stress, double strainRate, double taylorQuinney = 0.9
) {
    return taylorQuinney * stress * strainRate;
}

/**
 * @brief Calculate friction heat at tool-chip interface
 * 
 * q = τ · v
 * 
 * @param shearStress Friction shear stress (Pa)
 * @param slidingVelocity Relative sliding velocity (m/s)
 * @return Heat flux (W/m²)
 */
EP_HOST_DEVICE inline double calculateFrictionHeat(
    double shearStress, double slidingVelocity
) {
    return shearStress * slidingVelocity;
}

// ============================================================================
// Material Properties Database
// ============================================================================

/**
 * @brief Get Johnson-Cook parameters for common materials
 */
inline JohnsonCookParams getJCParams(const std::string& material) {
    JohnsonCookParams params;
    
    if (material == "Ti-6Al-4V" || material == "Ti64") {
        params.A = 880e6;
        params.B = 290e6;
        params.n = 0.47;
        params.C = 0.015;
        params.m = 1.0;
        params.T_melt = 1660.0;
    } else if (material == "AISI1045" || material == "1045Steel") {
        params.A = 553e6;
        params.B = 600e6;
        params.n = 0.234;
        params.C = 0.013;
        params.m = 1.0;
        params.T_melt = 1520.0;
    } else if (material == "Al7075" || material == "Aluminum7075") {
        params.A = 546e6;
        params.B = 678e6;
        params.n = 0.71;
        params.C = 0.024;
        params.m = 1.56;
        params.T_melt = 635.0;
    } else if (material == "Inconel718") {
        params.A = 1241e6;
        params.B = 622e6;
        params.n = 0.6522;
        params.C = 0.0134;
        params.m = 1.3;
        params.T_melt = 1336.0;
    }
    // Default is Ti-6Al-4V
    
    return params;
}

/**
 * @brief Get Usui wear parameters for tool materials
 */
inline UsuiWearParams getUsuiParams(const std::string& toolMaterial) {
    UsuiWearParams params;
    
    if (toolMaterial == "Carbide" || toolMaterial == "WC") {
        params.A = 1e-9;
        params.B = 1000.0;
    } else if (toolMaterial == "CBN") {
        params.A = 0.5e-9;
        params.B = 1200.0;
    } else if (toolMaterial == "HSS") {
        params.A = 5e-9;
        params.B = 800.0;
    }
    // Default is Carbide
    
    return params;
}

} // namespace edgepredict
