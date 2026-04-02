#pragma once
/**
 * @file GCodeInterpreter.h
 * @brief G-Code parsing and machine state interpolation
 */

#include "Types.h"
#include <string>
#include <vector>

namespace edgepredict {

/**
 * @brief Represents the machine state at a point in time
 */
struct MachineStateSnapshot {
    double time;            // Simulation time (seconds)
    Vec3 position;          // Tool center point (meters)
    Vec3 direction;         // Tool axis direction
    double feedRate;        // Current feed rate (m/s)
    double spindleRPM;      // Spindle speed
    bool spindleOn;         // Spindle is running
    bool coolantOn;         // Coolant is flowing
    
    // G-Code modal state
    bool absoluteMode;      // G90 (absolute) vs G91 (incremental)
    int motionMode;         // 0=rapid, 1=linear, 2=CW arc, 3=CCW arc
    
    MachineStateSnapshot() 
        : time(0), position(), direction(0, 0, 1),
          feedRate(0), spindleRPM(0), spindleOn(false), coolantOn(false),
          absoluteMode(true), motionMode(0) {}
};

/**
 * @brief Internal trajectory segment
 */
struct TrajectorySegment {
    double startTime;
    double endTime;
    Vec3 startPos;
    Vec3 endPos;
    double feedRate;
    double spindleRPM;
    int motionType;  // 0=rapid, 1=linear
};

/**
 * @brief G-Code interpreter for machine tool paths
 * 
 * Parses standard G-Code and provides time-based interpolation
 * of machine states for simulation.
 */
class GCodeInterpreter {
public:
    GCodeInterpreter();
    ~GCodeInterpreter();
    
    /**
     * @brief Load and parse G-Code file
     * @param path Path to G-Code file
     * @return true if parsing succeeded
     */
    bool loadFile(const std::string& path);
    
    /**
     * @brief Parse G-Code from string
     * @param gcodeContent G-Code content
     * @return true if parsing succeeded
     */
    bool parseString(const std::string& gcodeContent);
    
    /**
     * @brief Get machine state at simulation time
     * @param simTime Simulation time in seconds
     * @return Interpolated machine state
     */
    MachineStateSnapshot getStateAtTime(double simTime) const;
    
    /**
     * @brief Get total trajectory duration
     */
    double getTotalDuration() const { return m_totalDuration; }
    
    /**
     * @brief Get number of trajectory segments
     */
    size_t getSegmentCount() const { return m_segments.size(); }
    
    /**
     * @brief Check if G-Code is loaded
     */
    bool isLoaded() const { return m_isLoaded; }
    
    /**
     * @brief Get bounding box of entire toolpath
     */
    void getToolpathBounds(Vec3& minCorner, Vec3& maxCorner) const;
    
    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }
    
    /**
     * @brief Set rapid traverse rate (for G0 moves)
     * @param rate Rate in m/s
     */
    void setRapidRate(double rate) { m_rapidRate = rate; }
    
    /**
     * @brief Set unit conversion (1.0 for mm, 0.0254 for inches to mm)
     */
    void setUnitScale(double scale) { m_unitScale = scale; }

private:
    void parseLine(const std::string& line);
    void processGCommand(int gCode, double x, double y, double z, double f);
    void processMCommand(int mCode);
    void addSegment();
    
    std::vector<TrajectorySegment> m_segments;
    
    // Current parsing state
    MachineStateSnapshot m_currentState;
    Vec3 m_pendingTarget;
    double m_pendingFeed;
    
    // Configuration
    double m_rapidRate = 10.0;       // m/s (600 m/min)
    double m_unitScale = 0.001;      // mm to m
    
    // State
    bool m_isLoaded = false;
    double m_totalDuration = 0.0;
    std::string m_lastError;
};

} // namespace edgepredict
