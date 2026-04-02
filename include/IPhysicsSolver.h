#pragma once
/**
 * @file IPhysicsSolver.h
 * @brief Clean interface for physics solvers - NO circular dependencies
 * 
 * This interface is intentionally minimal. Solver implementations
 * include their own specific headers, not referenced here.
 */

#include <string>
#include <memory>

namespace edgepredict {

// Forward declarations only - no includes to avoid circular deps
class Config;

/**
 * @brief Abstract interface for all physics solvers
 * 
 * This clean interface allows dependency injection and easy testing.
 * Each solver implementation (SPH, FEM, CFD) implements this interface.
 */
class IPhysicsSolver {
public:
    virtual ~IPhysicsSolver() = default;
    
    /**
     * @brief Get solver name for logging
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Initialize solver with configuration
     * @param config Configuration object
     * @return true if initialization succeeded
     */
    virtual bool initialize(const Config& config) = 0;
    
    /**
     * @brief Advance simulation by one time step
     * @param dt Time step size (seconds)
     */
    virtual void step(double dt) = 0;
    
    /**
     * @brief Calculate stable time step based on CFL condition
     * @return Maximum stable time step (seconds)
     */
    virtual double getStableTimeStep() const = 0;
    
    /**
     * @brief Get current simulation time
     */
    virtual double getCurrentTime() const = 0;
    
    /**
     * @brief Get solver is initialized
     */
    virtual bool isInitialized() const = 0;
    
    /**
     * @brief Get axis-aligned bounding box of the solver's domain/particles
     */
    virtual void getBounds(double& minX, double& minY, double& minZ, 
                           double& maxX, double& maxY, double& maxZ) const = 0;
    
    /**
     * @brief Get number of nodes (for mesh-based solvers)
     */
    virtual int getNodeCount() const { return 0; }
    
    /**
     * @brief Get number of particles (for particle-based solvers)
     */
    virtual int getParticleCount() const { return 0; }
    
    /**
     * @brief Reset solver to initial state
     */
    virtual void reset() = 0;
};

/**
 * @brief Smart pointer type for solvers
 */
using ISolverPtr = std::unique_ptr<IPhysicsSolver>;

/**
 * @brief Interface for solvers that can provide metrics
 */
class IMetricsProvider {
public:
    virtual ~IMetricsProvider() = default;
    
    /**
     * @brief Get maximum stress in the system (Pa)
     */
    virtual double getMaxStress() const = 0;
    
    /**
     * @brief Get maximum temperature (°C)
     */
    virtual double getMaxTemperature() const = 0;
    
    /**
     * @brief Get total kinetic energy (J)
     */
    virtual double getTotalKineticEnergy() const = 0;

    /**
     * @brief Force synchronization of device-side metrics to host
     */
    virtual void syncMetrics() = 0;
};

/**
 * @brief Interface for thermal coupling between solvers
 */
class IThermalCoupling {
public:
    virtual ~IThermalCoupling() = default;
    
    /**
     * @brief Apply heat flux at a point
     * @param x, y, z Position
     * @param heatFlux Heat flux (W/m^2)
     */
    virtual void applyHeatFlux(double x, double y, double z, double heatFlux) = 0;
    
    /**
     * @brief Get temperature at a point
     * @param x, y, z Position
     * @return Temperature (°C)
     */
    virtual double getTemperatureAt(double x, double y, double z) const = 0;
};

} // namespace edgepredict
