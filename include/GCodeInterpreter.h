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
    double feedRate;
    double spindleRPM;
    int motionType;  // 0=rapid, 1=linear, 2=CW arc, 3=CCW arc
    
    // Linear segments
    Vec3 endPos;
    
    // Arc segments (G02/G03)
    bool isArc = false;
    Vec3 arcCenter;          // Center of arc (absolute)
    double arcRadius = 0;
    double arcStartAngle = 0;  // Start angle (radians)
    double arcEndAngle = 0;    // End angle (radians)
    double arcStartZ = 0;      // Z at start of arc
    double arcEndZ = 0;        // Z at end of arc
    int arcPlane = 0;          // 0=XY(G17), 1=XZ(G18), 2=YZ(G19)
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
    
    /**
     * @brief Advance the internal playhead by dt seconds
     * @param dt Time step in seconds
     * @return Machine state at the new time
     * 
     * This is the primary interface for the simulation step loop.
     * Each call advances the playhead and returns the exact position.
     */
    MachineStateSnapshot advanceByDt(double dt);
    
    /**
     * @brief Get instantaneous velocity vector at current playhead
     */
    Vec3 getVelocityVector() const;
    
    /**
     * @brief Get current motion mode (0=rapid, 1=linear, 2=CW arc, 3=CCW arc)
     */
    int getMotionMode() const { return m_currentState.motionMode; }
    
    /**
     * @brief Check if currently in rapid traverse (G00)
     */
    bool isRapid() const { return m_currentState.motionMode == 0; }
    
    /**
     * @brief Get active WCS index (0=G54, ..., 5=G59)
     */
    int getActiveWCS() const { return m_activeWCSIndex; }
    
    /**
     * @brief Reset playhead to start
     */
    void resetPlayhead() { m_playheadTime = 0.0; }

    /**
     * @brief Get the exact time where initial rapid moves end and first cutting move starts.
     *
     * Iterates through segments directly (not by time sampling) to find the
     * boundary between the initial G00 rapid sequence and the first G01/G02/G03.
     * This is used by the engine to pre-position the tool at clearance height.
     *
     * @return Time in seconds, or 0.0 if the first segment is already a cutting move
     */
    double getFirstCuttingMoveTime() const {
        if (m_segments.empty()) return 0.0;
        for (size_t i = 0; i < m_segments.size(); ++i) {
            if (m_segments[i].motionType != 0) {
                // Found the first non-rapid — return the START time of this segment
                // (which equals the END time of the last rapid)
                return m_segments[i].startTime;
            }
        }
        // All segments are rapids — return total duration
        return m_totalDuration;
    }

    /**
     * @brief Get the position at the end of a segment containing the given time.
     * Useful for getting exact endpoint positions (avoids floating-point interpolation errors).
     */
    Vec3 getSegmentEndPosition(double simTime) const {
        if (m_segments.empty()) return Vec3(0,0,0);
        for (const auto& seg : m_segments) {
            if (simTime >= seg.startTime && simTime <= seg.endTime) {
                return seg.endPos;
            }
        }
        return m_segments.back().endPos;
    }

private:
    void parseLine(const std::string& line);
    void processGCommand(int gCode, double x, double y, double z, 
                         double f, double iVal, double jVal, double kVal,
                         bool hasX, bool hasY, bool hasZ, 
                         bool hasI, bool hasJ, bool hasK);
    void processMCommand(int mCode);
    void addSegment();
    void addArcSegment(const Vec3& endPos, double iVal, double jVal, double kVal, bool clockwise);
    Vec3 interpolateArc(const TrajectorySegment& seg, double t) const;
    
    std::vector<TrajectorySegment> m_segments;
    
    // Current parsing state
    MachineStateSnapshot m_currentState;
    Vec3 m_pendingTarget;
    double m_pendingFeed;
    
    // Configuration
    double m_rapidRate = 10.0;       // m/s (600 m/min)
    double m_unitScale = 0.001;      // mm to m
    int m_arcPlane = 0;              // 0=XY(G17), 1=XZ(G18), 2=YZ(G19)
    int m_arcSegments = 36;          // Segments per full circle for arc interpolation
    
    // GCode state
    int m_activeWCSIndex = 0;        // 0=G54 through 5=G59
    
    // Playhead for advanceByDt()
    double m_playheadTime = 0.0;
    
    // State
    bool m_isLoaded = false;
    double m_totalDuration = 0.0;
    std::string m_lastError;
};

} // namespace edgepredict
