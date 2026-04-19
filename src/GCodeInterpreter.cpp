/**
 * @file GCodeInterpreter.cpp
 * @brief G-Code parsing implementation with true arc interpolation and G54-G59 support
 *
 * Digital Twin Foundation upgrade:
 *   - True G02/G03 circular interpolation (not linearized)
 *   - G54-G59 WCS switching
 *   - advanceByDt() for per-timestep tool position driving
 *   - G17/G18/G19 arc plane selection
 */

#include "GCodeInterpreter.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

GCodeInterpreter::GCodeInterpreter() {
    m_currentState = MachineStateSnapshot();
    m_pendingTarget = Vec3::zero();
    m_pendingFeed = 0.0;
}

GCodeInterpreter::~GCodeInterpreter() = default;

bool GCodeInterpreter::loadFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        m_lastError = "Cannot open G-Code file: " + path;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    bool result = parseString(buffer.str());
    
    if (result) {
        std::cout << "[GCodeInterpreter] Loaded: " << path << std::endl;
        std::cout << "  Segments: " << m_segments.size() << std::endl;
        std::cout << "  Duration: " << m_totalDuration << " seconds" << std::endl;
    }
    
    return result;
}

bool GCodeInterpreter::parseString(const std::string& gcodeContent) {
    m_segments.clear();
    m_currentState = MachineStateSnapshot();
    m_totalDuration = 0.0;
    m_isLoaded = false;
    m_playheadTime = 0.0;
    m_activeWCSIndex = 0;
    m_arcPlane = 0; // Default G17 (XY plane)
    
    std::istringstream stream(gcodeContent);
    std::string line;
    int lineNum = 0;
    
    while (std::getline(stream, line)) {
        lineNum++;
        
        // Remove comments (everything after ; or ())
        size_t commentPos = line.find(';');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        commentPos = line.find('(');
        if (commentPos != std::string::npos) {
            size_t closePos = line.find(')', commentPos);
            if (closePos != std::string::npos) {
                line = line.substr(0, commentPos) + line.substr(closePos + 1);
            }
        }
        
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            line = line.substr(0, end + 1);
        }
        
        if (line.empty()) continue;
        
        try {
            parseLine(line);
        } catch (const std::exception& e) {
            std::cerr << "[GCodeInterpreter] Warning at line " << lineNum 
                      << ": " << e.what() << std::endl;
        }
    }
    
    m_isLoaded = !m_segments.empty();
    return m_isLoaded;
}

void GCodeInterpreter::parseLine(const std::string& line) {
    // Convert to uppercase for parsing
    std::string upper = line;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    // Parse words: letter + number
    double x = m_currentState.position.x / m_unitScale;  // Keep in file units
    double y = m_currentState.position.y / m_unitScale;
    double z = m_currentState.position.z / m_unitScale;
    double f = m_pendingFeed;
    double iVal = 0, jVal = 0, kVal = 0;
    int gCode = -1;
    int mCode = -1;
    double sValue = m_currentState.spindleRPM;
    
    bool hasX = false, hasY = false, hasZ = false;
    bool hasI = false, hasJ = false, hasK = false;
    
    std::regex wordRegex("([GXYZFMSTHIJK])(-?[0-9]*\\.?[0-9]+)");
    std::sregex_iterator it(upper.begin(), upper.end(), wordRegex);
    std::sregex_iterator endIter;
    
    while (it != endIter) {
        std::smatch match = *it;
        char letter = match[1].str()[0];
        double value = std::stod(match[2].str());
        
        switch (letter) {
            case 'G': gCode = static_cast<int>(value); break;
            case 'X': x = value; hasX = true; break;
            case 'Y': y = value; hasY = true; break;
            case 'Z': z = value; hasZ = true; break;
            case 'F': f = value; break;
            case 'M': mCode = static_cast<int>(value); break;
            case 'S': sValue = value; break;
            case 'T': /* Tool number, ignored for now */ break;
            case 'I': iVal = value; hasI = true; break;
            case 'J': jVal = value; hasJ = true; break;
            case 'K': kVal = value; hasK = true; break;
            case 'H': /* Tool length offset register, ignored for now */ break;
        }
        
        ++it;
    }
    
    // Process M-codes first (spindle, coolant)
    if (mCode >= 0) {
        processMCommand(mCode);
    }
    
    // Update spindle
    m_currentState.spindleRPM = sValue;
    
    // Process G-codes
    if (gCode >= 0) {
        processGCommand(gCode, x, y, z, f, iVal, jVal, kVal,
                        hasX, hasY, hasZ, hasI, hasJ, hasK);
    } else if (hasX || hasY || hasZ) {
        // Motion without explicit G-code uses current motion mode
        processGCommand(m_currentState.motionMode, x, y, z, f, iVal, jVal, kVal,
                        hasX, hasY, hasZ, hasI, hasJ, hasK);
    }
}

void GCodeInterpreter::processGCommand(int gCode, double x, double y, double z, 
                                         double f, double iVal, double jVal, double kVal,
                                         bool hasX, bool hasY, bool hasZ,
                                         bool hasI, bool hasJ, bool hasK) {
    switch (gCode) {
        case 0: // Rapid move
            m_currentState.motionMode = 0;
            m_pendingTarget = Vec3(x, y, z) * m_unitScale;
            m_pendingFeed = m_rapidRate / m_unitScale;
            addSegment();
            break;
            
        case 1: // Linear interpolation
            m_currentState.motionMode = 1;
            m_pendingTarget = Vec3(x, y, z) * m_unitScale;
            m_pendingFeed = f / 60.0 * m_unitScale;  // Convert mm/min to m/s
            addSegment();
            break;
            
        case 2:  // CW arc
        case 3: { // CCW arc
            m_currentState.motionMode = gCode;
            Vec3 endPos = Vec3(x, y, z) * m_unitScale;
            double feedRate = f / 60.0 * m_unitScale;
            if (feedRate < 1e-12) feedRate = m_pendingFeed;
            m_pendingFeed = feedRate;
            
            addArcSegment(endPos, iVal * m_unitScale, jVal * m_unitScale, 
                          kVal * m_unitScale, gCode == 2);
            break;
        }
            
        case 17: // XY plane
            m_arcPlane = 0;
            break;
        case 18: // XZ plane  
            m_arcPlane = 1;
            break;
        case 19: // YZ plane
            m_arcPlane = 2;
            break;
            
        case 20: // Inches
            m_unitScale = 0.0254;
            break;
            
        case 21: // Millimeters
            m_unitScale = 0.001;
            break;
            
        case 28: // Home
            m_pendingTarget = Vec3::zero();
            m_pendingFeed = m_rapidRate;
            addSegment();
            break;
            
        // G54-G59: Work Coordinate System selection
        case 54: m_activeWCSIndex = 0; break;
        case 55: m_activeWCSIndex = 1; break;
        case 56: m_activeWCSIndex = 2; break;
        case 57: m_activeWCSIndex = 3; break;
        case 58: m_activeWCSIndex = 4; break;
        case 59: m_activeWCSIndex = 5; break;
            
        case 90: // Absolute positioning
            m_currentState.absoluteMode = true;
            break;
            
        case 91: // Incremental positioning
            m_currentState.absoluteMode = false;
            break;
            
        default:
            // Ignore unknown G-codes
            break;
    }
}

void GCodeInterpreter::processMCommand(int mCode) {
    switch (mCode) {
        case 3: // Spindle on CW
        case 4: // Spindle on CCW
            m_currentState.spindleOn = true;
            break;
            
        case 5: // Spindle off
            m_currentState.spindleOn = false;
            break;
            
        case 7: // Coolant on (mist)
        case 8: // Coolant on (flood)
            m_currentState.coolantOn = true;
            break;
            
        case 9: // Coolant off
            m_currentState.coolantOn = false;
            break;
            
        case 30: // End program
        case 2:  // End program
            break;
            
        default:
            break;
    }
}

void GCodeInterpreter::addSegment() {
    Vec3 startPos = m_currentState.position;
    Vec3 endPos = m_pendingTarget;
    
    if (!m_currentState.absoluteMode) {
        endPos = startPos + m_pendingTarget;
    }
    
    Vec3 delta = endPos - startPos;
    double distance = delta.length();
    
    if (distance < 1e-12) return;
    
    double feedRate = m_pendingFeed;
    if (feedRate < 1e-12) feedRate = 0.001;
    
    double duration = distance / feedRate;
    
    TrajectorySegment seg;
    seg.startTime = m_totalDuration;
    seg.endTime = m_totalDuration + duration;
    seg.startPos = startPos;
    seg.endPos = endPos;
    seg.feedRate = feedRate;
    seg.spindleRPM = m_currentState.spindleRPM;
    seg.motionType = m_currentState.motionMode;
    seg.isArc = false;
    
    m_segments.push_back(seg);
    
    m_totalDuration = seg.endTime;
    m_currentState.position = endPos;
    m_currentState.feedRate = feedRate;
    m_currentState.time = m_totalDuration;
}

void GCodeInterpreter::addArcSegment(const Vec3& endPos, double iVal, double jVal, double kVal, bool clockwise) {
    Vec3 startPos = m_currentState.position;
    
    // I/J/K are incremental offsets from start to center
    Vec3 center;
    double startAngle, endAngle, radius;
    
    // Determine arc geometry based on active plane
    switch (m_arcPlane) {
        case 0: { // G17 - XY plane
            center = Vec3(startPos.x + iVal, startPos.y + jVal, startPos.z);
            
            double startDx = startPos.x - center.x;
            double startDy = startPos.y - center.y;
            double endDx = endPos.x - center.x;
            double endDy = endPos.y - center.y;
            
            radius = std::sqrt(startDx * startDx + startDy * startDy);
            startAngle = std::atan2(startDy, startDx);
            endAngle = std::atan2(endDy, endDx);
            break;
        }
        case 1: { // G18 - XZ plane
            center = Vec3(startPos.x + iVal, startPos.y, startPos.z + kVal);
            
            double startDx = startPos.x - center.x;
            double startDz = startPos.z - center.z;
            double endDx = endPos.x - center.x;
            double endDz = endPos.z - center.z;
            
            radius = std::sqrt(startDx * startDx + startDz * startDz);
            startAngle = std::atan2(startDz, startDx);
            endAngle = std::atan2(endDz, endDx);
            break;
        }
        case 2: { // G19 - YZ plane
            center = Vec3(startPos.x, startPos.y + jVal, startPos.z + kVal);
            
            double startDy = startPos.y - center.y;
            double startDz = startPos.z - center.z;
            double endDy = endPos.y - center.y;
            double endDz = endPos.z - center.z;
            
            radius = std::sqrt(startDy * startDy + startDz * startDz);
            startAngle = std::atan2(startDz, startDy);
            endAngle = std::atan2(endDz, endDy);
            break;
        }
        default:
            return; // Invalid plane
    }
    
    if (radius < 1e-12) return; // Degenerate arc
    
    // Calculate sweep angle
    double sweep = endAngle - startAngle;
    
    if (clockwise) { // G02
        if (sweep >= 0) sweep -= 2.0 * PI;
        // Full circle if start == end
        if (std::abs(sweep) < 1e-9) sweep = -2.0 * PI;
    } else { // G03
        if (sweep <= 0) sweep += 2.0 * PI;
        if (std::abs(sweep) < 1e-9) sweep = 2.0 * PI;
    }
    
    // Arc length
    double arcLength = std::abs(sweep) * radius;
    
    // Include linear Z movement in distance calculation (helical interpolation)
    double dz = 0;
    if (m_arcPlane == 0) dz = endPos.z - startPos.z; // XY plane, Z moves linearly
    else if (m_arcPlane == 1) dz = endPos.y - startPos.y; // XZ plane, Y moves linearly
    else dz = endPos.x - startPos.x; // YZ plane, X moves linearly
    
    double totalLength = std::sqrt(arcLength * arcLength + dz * dz);
    
    double feedRate = m_pendingFeed;
    if (feedRate < 1e-12) feedRate = 0.001;
    double duration = totalLength / feedRate;
    
    TrajectorySegment seg;
    seg.startTime = m_totalDuration;
    seg.endTime = m_totalDuration + duration;
    seg.startPos = startPos;
    seg.endPos = endPos;
    seg.feedRate = feedRate;
    seg.spindleRPM = m_currentState.spindleRPM;
    seg.motionType = clockwise ? 2 : 3;
    seg.isArc = true;
    seg.arcCenter = center;
    seg.arcRadius = radius;
    seg.arcStartAngle = startAngle;
    seg.arcEndAngle = startAngle + sweep; // Not normalized — preserves direction
    seg.arcPlane = m_arcPlane;
    
    // Store linear axis values for helical interpolation
    if (m_arcPlane == 0) { seg.arcStartZ = startPos.z; seg.arcEndZ = endPos.z; }
    else if (m_arcPlane == 1) { seg.arcStartZ = startPos.y; seg.arcEndZ = endPos.y; }
    else { seg.arcStartZ = startPos.x; seg.arcEndZ = endPos.x; }
    
    m_segments.push_back(seg);
    
    m_totalDuration = seg.endTime;
    m_currentState.position = endPos;
    m_currentState.feedRate = feedRate;
    m_currentState.time = m_totalDuration;
}

Vec3 GCodeInterpreter::interpolateArc(const TrajectorySegment& seg, double t) const {
    double angle = seg.arcStartAngle + (seg.arcEndAngle - seg.arcStartAngle) * t;
    double linearZ = seg.arcStartZ + (seg.arcEndZ - seg.arcStartZ) * t;
    
    switch (seg.arcPlane) {
        case 0: // XY plane
            return Vec3(
                seg.arcCenter.x + seg.arcRadius * std::cos(angle),
                seg.arcCenter.y + seg.arcRadius * std::sin(angle),
                linearZ
            );
        case 1: // XZ plane  
            return Vec3(
                seg.arcCenter.x + seg.arcRadius * std::cos(angle),
                linearZ,
                seg.arcCenter.z + seg.arcRadius * std::sin(angle)
            );
        case 2: // YZ plane
            return Vec3(
                linearZ,
                seg.arcCenter.y + seg.arcRadius * std::cos(angle),
                seg.arcCenter.z + seg.arcRadius * std::sin(angle)
            );
        default:
            return seg.startPos;
    }
}

MachineStateSnapshot GCodeInterpreter::getStateAtTime(double simTime) const {
    MachineStateSnapshot state;
    
    if (m_segments.empty()) {
        return state;
    }
    
    // Clamp to valid range
    if (simTime <= 0) {
        state.position = m_segments.front().startPos;
        state.feedRate = m_segments.front().feedRate;
        state.spindleRPM = m_segments.front().spindleRPM;
        state.time = 0;
        state.motionMode = m_segments.front().motionType;
        return state;
    }
    
    if (simTime >= m_totalDuration) {
        state.position = m_segments.back().endPos;
        state.feedRate = m_segments.back().feedRate;
        state.spindleRPM = m_segments.back().spindleRPM;
        state.time = m_totalDuration;
        state.motionMode = m_segments.back().motionType;
        return state;
    }
    
    // Binary search for segment
    size_t left = 0, right = m_segments.size() - 1;
    while (left < right) {
        size_t mid = (left + right) / 2;
        if (m_segments[mid].endTime <= simTime) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    const TrajectorySegment& seg = m_segments[left];
    
    // Parametric t within segment
    double t = (simTime - seg.startTime) / (seg.endTime - seg.startTime);
    t = std::max(0.0, std::min(1.0, t));
    
    // Interpolate position
    if (seg.isArc) {
        state.position = interpolateArc(seg, t);
    } else {
        state.position = seg.startPos + (seg.endPos - seg.startPos) * t;
    }
    
    state.feedRate = seg.feedRate;
    state.spindleRPM = seg.spindleRPM;
    state.time = simTime;
    state.spindleOn = seg.spindleRPM > 0;
    state.motionMode = seg.motionType;
    
    // Calculate direction of motion
    if (seg.isArc) {
        // Tangent direction on arc
        double angle = seg.arcStartAngle + (seg.arcEndAngle - seg.arcStartAngle) * t;
        double sweepDir = (seg.arcEndAngle > seg.arcStartAngle) ? 1.0 : -1.0;
        
        switch (seg.arcPlane) {
            case 0: // XY plane
                state.direction = Vec3(
                    -seg.arcRadius * std::sin(angle) * sweepDir,
                    seg.arcRadius * std::cos(angle) * sweepDir,
                    0
                ).normalized();
                break;
            case 1: // XZ plane
                state.direction = Vec3(
                    -seg.arcRadius * std::sin(angle) * sweepDir,
                    0,
                    seg.arcRadius * std::cos(angle) * sweepDir
                ).normalized();
                break;
            case 2: // YZ plane
                state.direction = Vec3(
                    0,
                    -seg.arcRadius * std::sin(angle) * sweepDir,
                    seg.arcRadius * std::cos(angle) * sweepDir
                ).normalized();
                break;
        }
    } else {
        Vec3 delta = seg.endPos - seg.startPos;
        if (delta.lengthSq() > 1e-12) {
            state.direction = delta.normalized();
        }
    }
    
    return state;
}

MachineStateSnapshot GCodeInterpreter::advanceByDt(double dt) {
    m_playheadTime += dt;
    return getStateAtTime(m_playheadTime);
}

Vec3 GCodeInterpreter::getVelocityVector() const {
    if (m_segments.empty()) return Vec3::zero();
    
    auto state = getStateAtTime(m_playheadTime);
    return state.direction * state.feedRate;
}

void GCodeInterpreter::getToolpathBounds(Vec3& minCorner, Vec3& maxCorner) const {
    if (m_segments.empty()) {
        minCorner = Vec3::zero();
        maxCorner = Vec3::zero();
        return;
    }
    
    minCorner = m_segments[0].startPos;
    maxCorner = m_segments[0].startPos;
    
    for (const auto& seg : m_segments) {
        // Check start and end
        minCorner.x = std::min({minCorner.x, seg.startPos.x, seg.endPos.x});
        minCorner.y = std::min({minCorner.y, seg.startPos.y, seg.endPos.y});
        minCorner.z = std::min({minCorner.z, seg.startPos.z, seg.endPos.z});
        maxCorner.x = std::max({maxCorner.x, seg.startPos.x, seg.endPos.x});
        maxCorner.y = std::max({maxCorner.y, seg.startPos.y, seg.endPos.y});
        maxCorner.z = std::max({maxCorner.z, seg.startPos.z, seg.endPos.z});
        
        // For arcs, also check extremes
        if (seg.isArc) {
            // Sample arc at multiple points to capture bounds
            for (int i = 0; i <= 36; ++i) {
                double t = static_cast<double>(i) / 36.0;
                Vec3 pt = interpolateArc(seg, t);
                minCorner.x = std::min(minCorner.x, pt.x);
                minCorner.y = std::min(minCorner.y, pt.y);
                minCorner.z = std::min(minCorner.z, pt.z);
                maxCorner.x = std::max(maxCorner.x, pt.x);
                maxCorner.y = std::max(maxCorner.y, pt.y);
                maxCorner.z = std::max(maxCorner.z, pt.z);
            }
        }
    }
}

} // namespace edgepredict
