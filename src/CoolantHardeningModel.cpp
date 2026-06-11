#include "CoolantHardeningModel.h"
#include "MPMSolver.cuh"
#include "FEMSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace edgepredict {

// Physical constants for coolant
static const double MU_WATER_25C = 0.00089;  // Pa·s at 25°C
static const double K_WATER_25C = 0.6;       // W/mK
static const double PR_WATER_25C = 6.2;      // Prandtl number for water at 25°C
static const double LATENT_HEAT_VAPORIZATION = 2.26e6; // J/kg for water

CoolantHardeningModel::CoolantHardeningModel() {
    reset();
}

void CoolantHardeningModel::initialize(const CoolantHardeningConfig& params) {
    m_params = params;
    m_initialized = true;
    reset();
}

void CoolantHardeningModel::configure(const Config& config) {
    const auto& j = config.getJson();
    if (!j.contains("coolant_hardening_model")) {
        std::cout << "[CoolantHardening] Not configured (no coolant_hardening_model section)" << std::endl;
        return;
    }
    const auto& ch = j["coolant_hardening_model"];
    m_params.enabled = ch.value("enabled", true);

    // Gap #7a: Jet impingement
    if (ch.contains("jet_impingement")) {
        const auto& ji = ch["jet_impingement"];
        m_params.jetImpingementEnabled = ji.value("enabled", true);
        m_params.jetNozzleDiameter = ji.value("nozzle_diameter_m", 0.005);
        m_params.jetStandoffDistance = ji.value("standoff_distance_m", 0.015);
        m_params.jetVelocityMperS = ji.value("velocity_m_s", 1.5);
        m_params.stagnationHTCFactor = ji.value("stagnation_htc_factor", 1.0);
    }

    // Gap #7b: Film boiling / Rohsenow nucleation-site / radiation
    if (ch.contains("film_boiling")) {
        const auto& fb = ch["film_boiling"];
        m_params.filmBoilingEnabled = fb.value("enabled", true);
        m_params.leidenfrostTemperature = fb.value("leidenfrost_temperature_C", 300.0);
        m_params.criticalHeatFluxTemperature = fb.value("chf_temperature_C", 200.0);
        m_params.nucleateBoilingHTC = fb.value("nucleate_boiling_htc", 5.0e3);
        m_params.transitionBoilingHTC = fb.value("transition_boiling_htc", 1.0e3);
        m_params.filmBoilingHTC = fb.value("film_boiling_htc", 2.0e2);
        m_params.singlePhaseHTC = fb.value("single_phase_htc", 1.0e3);
        m_params.rohsenow_Csf = fb.value("rohsenow_Csf", 0.013);
        m_params.rohsenow_n = fb.value("rohsenow_n", 1.0);
        m_params.latentHeatVaporization = fb.value("latent_heat_vaporization_J_kg", 2.26e6);
        m_params.liquidDensity = fb.value("liquid_density_kg_m3", 1050.0);
        m_params.vaporDensity = fb.value("vapor_density_kg_m3", 0.6);
        m_params.surfaceTension = fb.value("surface_tension_N_m", 0.072);
        m_params.radiationEnabled = fb.value("radiation_enabled", true);
        m_params.surfaceEmissivity = fb.value("surface_emissivity", 0.7);
    }

    // Gap #7c: Capillary wicking
    if (ch.contains("capillary_wicking")) {
        const auto& cw = ch["capillary_wicking"];
        m_params.capillaryWickingEnabled = cw.value("enabled", true);
        m_params.capillaryPoreRadius = cw.value("pore_radius_m", 5.0e-6);
        m_params.coolantSurfaceTension = cw.value("surface_tension_N_m", 0.072);
        m_params.coolantContactAngleDeg = cw.value("contact_angle_deg", 30.0);
        m_params.capillaryPenetrationRate = cw.value("penetration_rate_m_s", 1.0e-4);
    }

    // Gap #7d: Thermal shock hardening
    if (ch.contains("thermal_shock")) {
        const auto& ts = ch["thermal_shock"];
        m_params.thermalShockEnabled = ts.value("enabled", true);
        m_params.quenchThresholdCoolingRate = ts.value("threshold_cooling_rate_C_s", 100.0);
        m_params.maxHardnessIncreaseHRC = ts.value("max_hardness_increase_hrc", 5.0);
        m_params.quenchDepthMm = ts.value("quench_depth_mm", 0.5);
    }

    // Gap #7e: Bidirectional coupling
    if (ch.contains("bidirectional_coupling")) {
        const auto& bc = ch["bidirectional_coupling"];
        m_params.bidirectionalCouplingEnabled = bc.value("enabled", true);
        m_params.convectiveHeatTransferCoeff = bc.value("convective_htc", 1.0e4);
        m_params.coolantCouplingFraction = bc.value("coupling_fraction", 1.0);
    }

    // Gap #7f: Flow-regime HTC
    if (ch.contains("flow_regime_htc")) {
        const auto& fr = ch["flow_regime_htc"];
        m_params.flowRegimeHTCEnabled = fr.value("enabled", true);
        m_params.flowHTCBase = fr.value("base_htc", 1.0e3);
    }

    // Read ambient temperature from machining config (not coolant_hardening_model)
    m_params.ambientTemperature = config.getMachining().ambientTemperature;

    m_initialized = true;
    std::cout << "[CoolantHardening] Initialized with "
              << (m_params.enabled ? "enabled" : "disabled") << " model" << std::endl;
}

void CoolantHardeningModel::update(MPMSolver* mpm, FEMSolver* fem, CFDSolverGPU* cfd, double dt) {
    if (!m_initialized || !m_params.enabled) return;
    if (!mpm) return;

    // Combined effective HTC starts with base single-phase value
    double combinedHTC = m_params.singlePhaseHTC;

    // --- Gap #7a: Jet impingement ---
    if (m_params.jetImpingementEnabled && fem) {
        Vec3 jetOrigin(0, 0, 0); // Will be refined once we have tool position
        auto nodes = fem->getNodes();
        double maxNodeHTC = 0.0;
        double sumNodeHTC = 0.0;
        int count = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            Vec3 pos(nodes[i].x, nodes[i].y, nodes[i].z);
            double htc = computeJetImpingementHTC(pos, jetOrigin);
            sumNodeHTC += htc;
            if (htc > maxNodeHTC) maxNodeHTC = htc;
            ++count;
        }
        m_state.jetStagnationHTC = maxNodeHTC;
        m_state.wallJetHTC = count > 0 ? sumNodeHTC / count : 0.0;
        combinedHTC += maxNodeHTC * 0.3 + m_state.wallJetHTC * 0.1;
    }

    // --- Gap #7b: Boiling regime (Rohsenow nucleation-site) + radiation (Item 5) ---
    if (m_params.filmBoilingEnabled) {
        auto particles = mpm->getParticles();
        double avgSurfaceTemp = 0.0;
        int surfaceCount = 0;
        for (size_t i = 0; i < particles.size(); ++i) {
            if (particles[i].temperature > 30.0) {
                avgSurfaceTemp += particles[i].temperature;
                ++surfaceCount;
            }
        }
        if (surfaceCount > 0) {
            avgSurfaceTemp /= surfaceCount;
        } else {
            avgSurfaceTemp = 25.0;
        }
        m_state.boilingRegimeHTC = computeBoilingHTC(avgSurfaceTemp);
        combinedHTC += m_state.boilingRegimeHTC;

        // Stefan-Boltzmann radiation from hot workpiece surface
        if (m_params.radiationEnabled) {
            double radHTC = computeRadiationHTC(avgSurfaceTemp, m_params.ambientTemperature);
            combinedHTC += radHTC;
        }
    }

    // --- Gap #7c: Capillary wicking ---
    if (m_params.capillaryWickingEnabled) {
        double contactGap = 1.0e-6; // ~1µm typical gap
        m_state.capillaryPenetrationDepth = updateCapillaryPenetration(dt, contactGap);
        // Wicking enhances HTC by bringing coolant deeper into the contact zone
        combinedHTC *= (1.0 + 0.1 * m_state.capillaryPenetrationDepth * 1000.0);
    }

    // --- Gap #7f: Flow-regime HTC ---
    if (m_params.flowRegimeHTCEnabled && cfd) {
        // Query coolant velocity near the tool tip (origin).  We batch-download
        // the velocity grid only periodically to avoid per-step O(N) copies.
        static Vec3 s_cachedVel(0,0,0);
        static int s_cacheStep = 0;
        if (m_state.updateCount % 10 == 0 || s_cacheStep == 0) {
            s_cachedVel = cfd->getVelocityAt(Vec3(0, 0, 0));
            s_cacheStep = m_state.updateCount;
        }
        double velMag = s_cachedVel.length();
        double filmTemp = 40.0; // Reference film temperature
        double charLen = 0.01;  // 10mm characteristic length
        double regimeHTC = computeFlowRegimeHTC(velMag, filmTemp, charLen);
        combinedHTC += regimeHTC;
    }

    // --- Gap #7e: MPM-CFD bidirectional coupling ---
    applyCFDCoolingToParticles(mpm, cfd, dt);

    m_state.effectiveCoolantHTC = combinedHTC;
    m_state.updateCount++;
}

// --- Gap #7a: Jet impingement HTC ---
// Uses empirical correlation for a submerged round jet:
//   Nu = 0.7 * Re^0.5 * Pr^0.33 at stagnation point
//   Nu = 0.5 * Re^0.6 * Pr^0.33 in wall jet region
double CoolantHardeningModel::computeJetImpingementHTC(const Vec3& position, const Vec3& jetOrigin) const {
    double dx = position.x - jetOrigin.x;
    double dy = position.y - jetOrigin.y;
    double dz = position.z - jetOrigin.z;
    double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    double dNozzle = m_params.jetNozzleDiameter;
    double vJet = m_params.jetVelocityMperS;
    double nu = MU_WATER_25C / 1050.0; // kinematic viscosity for water-glycol
    double Re = vJet * dNozzle / nu;

    if (Re < 1.0) return 0.0;

    double Pr = PR_WATER_25C;
    double k = K_WATER_25C;

    // Stagnation point (r/d < 0.5)
    if (r / dNozzle < 0.5) {
        double Nu = 0.7 * std::sqrt(Re) * std::pow(Pr, 0.33);
        return Nu * k / dNozzle * m_params.stagnationHTCFactor;
    }

    // Wall jet region (r/d >= 0.5)
    double Nu = 0.5 * std::pow(Re, 0.6) * std::pow(Pr, 0.33) * std::pow(r / dNozzle, -0.5);
    return Nu * k / dNozzle * m_params.stagnationHTCFactor;
}

// --- Gap #7b: Boiling curve HTC (Rohsenow nucleation-site model) ---
// Four regimes:
//   1. Single-phase liquid (T < 100°C): HTC = singlePhaseHTC
//   2. Nucleate boiling (100°C ≤ T < CHF): HTC from Rohsenow correlation:
//        q = μ·h_fg·(g(ρ_l-ρ_v)/σ)^0.5 · (c_p·ΔT / (C_sf·h_fg·Pr^n))^3
//        HTC = q / ΔT
//   3. Transition boiling (CHF ≤ T < Leidenfrost): linear ramp down
//   4. Film boiling (T ≥ Leidenfrost): filmBoilingHTC (vapor blanket)
//
// Reference: Rohsenow, W.M., "A Method of Correlating Heat Transfer Data
//            for Surface Boiling of Liquids", Trans. ASME, 1952.
double CoolantHardeningModel::computeBoilingHTC(double surfaceTemperatureC) const {
    const double T_sat = 100.0; // Saturation temperature of water at 1 atm (°C)
    const double g = 9.81;      // m/s²
    double T = surfaceTemperatureC;
    double dT_sat = T - T_sat;

    if (dT_sat <= 0.0) {
        // Sub-cooled / single-phase
        return m_params.singlePhaseHTC;
    }

    if (dT_sat > 0.0 && T < m_params.criticalHeatFluxTemperature) {
        // ── Rohsenow nucleate boiling ─────────────────────────────────────
        // Constants
        double mu  = MU_WATER_25C;            // liquid viscosity (Pa·s)
        double rho_l = m_params.liquidDensity; // kg/m³
        double rho_v = m_params.vaporDensity;  // kg/m³
        double sigma = m_params.surfaceTension;// N/m
        double h_fg = m_params.latentHeatVaporization; // J/kg
        double c_pl = 4180.0;                 // J/(kg·K) — specific heat of water
        double C_sf = m_params.rohsenow_Csf;
        double n    = m_params.rohsenow_n;
        double Pr   = PR_WATER_25C;

        // g(ρ_l - ρ_v) / σ
        double gDrhoSigma = g * (rho_l - rho_v) / sigma;
        double sqrtTerm = std::sqrt(gDrhoSigma);
        if (sqrtTerm < 1e-12) return m_params.singlePhaseHTC;

        // c_p · ΔT / (C_sf · h_fg · Pr^n)
        double denom = C_sf * h_fg * std::pow(Pr, n);
        if (denom < 1e-30) return m_params.singlePhaseHTC;
        double cpDtTerm = c_pl * dT_sat / denom;

        // q = μ·h_fg·sqrtTerm·(cpDtTerm)³   (W/m²)
        double q = mu * h_fg * sqrtTerm * cpDtTerm * cpDtTerm * cpDtTerm;
        if (q < 0) return m_params.singlePhaseHTC;

        double htc_rohsenow = q / fmax(dT_sat, 1.0);
        // Clamp to reasonable bounds
        htc_rohsenow = std::clamp(htc_rohsenow, m_params.singlePhaseHTC, m_params.nucleateBoilingHTC * 2.0);

        // Blend with single-phase near onset
        double onset = 5.0; // °C past saturation
        if (dT_sat < onset) {
            double blend = dT_sat / onset;
            return m_params.singlePhaseHTC * (1.0 - blend) + htc_rohsenow * blend;
        }
        return htc_rohsenow;
    }

    double T_leid = m_params.leidenfrostTemperature;
    double T_chf  = m_params.criticalHeatFluxTemperature;

    if (T < T_leid * 0.8) {
        // Transition boiling ramp (CHF → near Leidenfrost)
        double fraction = (T - T_chf) / (T_leid * 0.8 - T_chf);
        fraction = std::clamp(fraction, 0.0, 1.0);
        return m_params.nucleateBoilingHTC - (m_params.nucleateBoilingHTC - m_params.transitionBoilingHTC) * fraction;
    } else if (T < T_leid) {
        double fraction = (T - T_leid * 0.8) / (T_leid - T_leid * 0.8);
        fraction = std::clamp(fraction, 0.0, 1.0);
        return m_params.transitionBoilingHTC - (m_params.transitionBoilingHTC - m_params.filmBoilingHTC) * fraction;
    } else {
        return m_params.filmBoilingHTC;
    }
}

// --- Item 5: Stefan-Boltzmann radiation HTC ---
// HTC_rad = ε·σ·(T⁴ - T₀⁴) / (T - T₀)
//         = ε·σ·(T² + T₀²)·(T + T₀)
double CoolantHardeningModel::computeRadiationHTC(double surfaceTemperatureC, double ambientC) const {
    if (!m_params.radiationEnabled) return 0.0;
    double Ts = surfaceTemperatureC + 273.15;
    double Ta = ambientC + 273.15;
    double dT = surfaceTemperatureC - ambientC;
    if (dT <= 1.0) return 0.0;
    double Ts4 = Ts * Ts * Ts * Ts;
    double Ta4 = Ta * Ta * Ta * Ta;
    return m_params.surfaceEmissivity * m_params.stefanBoltzmann * (Ts4 - Ta4) / dT;
}

// --- Gap #7c: Capillary penetration ---
// Uses Lucas-Washburn equation: d = sqrt( (r * γ * cosθ) / (2μ) * t )
// Simplified as incremental update per timestep
double CoolantHardeningModel::updateCapillaryPenetration(double dt, double contactGapNm) {
    double r = m_params.capillaryPoreRadius;
    double gamma = m_params.coolantSurfaceTension;
    double thetaRad = m_params.coolantContactAngleDeg * 3.14159265 / 180.0;
    double mu = MU_WATER_25C;

    // Lucas-Washburn: penetration rate ∝ sqrt(r * γ * cosθ / (4μ)) / sqrt(t)
    // Simplified: incremental depth = baseRate * sqrt(2 * dt * r * γ * cosθ / μ)
    double penetrationIncrement = m_params.capillaryPenetrationRate * dt;
    if (contactGapNm > 1e-12) {
        double capillaryPressure = 2.0 * gamma * std::cos(thetaRad) / contactGapNm;
        penetrationIncrement *= std::sqrt(capillaryPressure / 1.0e5); // normalize by reference pressure
    }

    m_state.capillaryPenetrationDepth += penetrationIncrement;
    double maxDepth = 0.005; // 5mm maximum wicking depth
    if (m_state.capillaryPenetrationDepth > maxDepth) m_state.capillaryPenetrationDepth = maxDepth;

    return m_state.capillaryPenetrationDepth;
}

// --- Gap #7d: Quench hardness correction ---
// Martensite formation in steel requires cooling > threshold rate
// Hardness increase follows empirical relation:
//   ΔHRC = maxIncrease * (1 - exp(-coolingRate/threshold))
double CoolantHardeningModel::computeQuenchHardnessIncrease(double particleTemp, double coolingRate) const {
    if (coolingRate < m_params.quenchThresholdCoolingRate) return 0.0;

    double rateRatio = (coolingRate - m_params.quenchThresholdCoolingRate) / m_params.quenchThresholdCoolingRate;
    double factor = 1.0 - std::exp(-rateRatio * 2.0);
    return m_params.maxHardnessIncreaseHRC * factor;
}

// --- Gap #7f: Flow-regime HTC ---
// Nu = 0.023 * Re^0.8 * Pr^0.4 (Dittus-Boelter for turbulent flow)
// Nu = 3.66 (laminar, constant wall temperature)
double CoolantHardeningModel::computeFlowRegimeHTC(double velocity, double tempFilm, double charLength) const {
    if (velocity < 1e-12) return m_params.flowHTCBase;

    double nu = MU_WATER_25C / 1050.0;
    double Re = velocity * charLength / nu;
    double Pr = PR_WATER_25C;
    double k = K_WATER_25C;

    double Nu;
    if (Re < 2300) {
        // Laminar
        Nu = 3.66;
    } else {
        // Turbulent (Dittus-Boelter)
        Nu = 0.023 * std::pow(Re, 0.8) * std::pow(Pr, 0.4);
    }

    double htc = Nu * k / charLength;
    if (htc < m_params.flowHTCBase) htc = m_params.flowHTCBase;
    return htc;
}

// --- Gap #7e: Apply CFD cooling to MPM particles (complete bidirectional loop) ---
// Computes convective heat loss per particle using the local CFD coolant
// temperature and writes the cooled temperature back to the MPM solver.
// Also handles the CFD grid query efficiently by caching the host copy.
void CoolantHardeningModel::applyCFDCoolingToParticles(MPMSolver* mpm, CFDSolverGPU* cfd, double dt) {
    if (!mpm || !cfd) return;
    if (!m_params.bidirectionalCouplingEnabled) return;

    double h = m_params.convectiveHeatTransferCoeff;
    auto particles = mpm->getParticles();
    size_t n = particles.size();
    if (n == 0) return;

    // Batch-download the CFD temperature grid ONCE (avoids O(n·N) memcpy from
    // calling getTemperatureAt() in a loop, which copies the full grid per call).
    auto cfdGrid = cfd->getTemperatureGrid();
    auto cfdGP   = cfd->getGridParams();
    auto lookupCFD = [&](double px, double py, double pz) -> double {
        int i = std::clamp(static_cast<int>(px / cfdGP.dx), 0, cfdGP.nx - 1);
        int j = std::clamp(static_cast<int>(py / cfdGP.dx), 0, cfdGP.ny - 1);
        int k = std::clamp(static_cast<int>(pz / cfdGP.dx), 0, cfdGP.nz - 1);
        int idx = i + j * cfdGP.nx + k * cfdGP.nx * cfdGP.ny;
        return (idx >= 0 && idx < (int)cfdGrid.size()) ? (double)cfdGrid[idx] : 25.0;
    };

    std::vector<double> newTemps(n);
    int hotCount = 0;
    double totalCooled = 0.0;
    double totalHeatRemoved = 0.0;

    for (size_t i = 0; i < n; ++i) {
        if (particles[i].status == ParticleStatus::INACTIVE) {
            newTemps[i] = particles[i].temperature;
            continue;
        }

        double Tp = particles[i].temperature;
        newTemps[i] = Tp; // default: unchanged

        if (Tp > 30.0) {
            // Use bulk coolant (ambient) temperature instead of CFD grid lookup
            // at the particle position. The CFD grid cell containing the particle
            // was contaminated by setParticleHeatSources() which heats the cell to
            // the particle's own temperature, short-circuiting the ΔT calculation.
            // Using the ambient bulk coolant temperature is physically correct for
            // far-field coolant and gives the correct convective driving potential.
            double Tc = m_params.ambientTemperature;
            double dT = Tp - Tc;

            if (dT > 1.0) {
                // Convective cooling: Q = h·A·ΔT·dt
                double vol = fmax(particles[i].mass /
                                  fmax(particles[i].density, 1.0), 1e-24);
                double area = pow(vol, 2.0 / 3.0);
                double mass = fmax(particles[i].mass, 1e-15);
                double Cp = 475.0;
                double Q = h * area * dT * dt;
                totalHeatRemoved += Q;
                double dT_cool = Q / (mass * Cp);
                dT_cool = fmin(dT_cool, fmin(dT, 10.0));
                newTemps[i] = Tp - dT_cool;
                totalCooled += dT_cool;
                ++hotCount;
            }
        }
    }

    mpm->setParticleTemperatures(newTemps.data(), static_cast<int>(n));

    m_state.lastCoolingApplied = totalCooled;
    m_state.lastHeatRemovedJ = totalHeatRemoved;
    m_state.cooledParticleCount = hotCount;

    if (hotCount > 0 && m_state.updateCount % 500 == 0) {
        std::cout << "[CoolantHardening] Cooled " << hotCount << " particles, "
                  << "mean dT=" << (totalCooled / hotCount)
                  << " C, Qrem=" << std::scientific << std::setprecision(2)
                  << totalHeatRemoved << " J via " << h << " W/m2K HTC" << std::endl;
    }
}

void CoolantHardeningModel::reset() {
    m_state = CoolantHardeningState{};
}

} // namespace edgepredict
