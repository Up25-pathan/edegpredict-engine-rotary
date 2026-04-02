/**
 * @file GCodeInterpreter.cpp
 * @brief G-Code parsing implementation
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
    int gCode = -1;
    int mCode = -1;
    double sValue = m_currentState.spindleRPM;
    
    bool hasX = false, hasY = false, hasZ = false;
    
    std::regex wordRegex("([GXYZFMST])(-?[0-9]*\\.?[0-9]+)");
    std::sregex_iterator it(upper.begin(), upper.end(), wordRegex);
    std::sregex_iterator end;
    
    while (it != end) {
        std::smatch match = *it;
        char letter = match[1].str()[0];
        double value = std::stod(match[2].str());
        
        switch (letter) {
            case 'G': gCode = static_cast<int>(value); break;
            case 'X': x = value; hasX = true; break;
            case 'Y': y = value; hasY = true; break;
            case 'Z': z = value; hasZ = true; break;
            case 'F': f = value; break;  // mm/min typically
            case 'M': mCode = static_cast<int>(value); break;
            case 'S': sValue = value; break;
            case 'T': /* Tool number, ignored for now */ break;
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
        processGCommand(gCode, x, y, z, f);
    } else if (hasX || hasY || hasZ) {
        // Motion without explicit G-code uses current motion mode
        processGCommand(m_currentState.motionMode, x, y, z, f);
    }
}

void GCodeInterpreter::processGCommand(int gCode, double x, double y, double z, double f) {
    switch (gCode) {
        case 0: // Rapid move
            m_currentState.motionMode = 0;
            m_pendingTarget = Vec3(x, y, z) * m_unitScale;
            m_pendingFeed = m_rapidRate / m_unitScale;  // Use rapid rate
            addSegment();
            break;
            
        case 1: // Linear interpolation
            m_currentState.motionMode = 1;
            m_pendingTarget = Vec3(x, y, z) * m_unitScale;
            m_pendingFeed = f / 60.0 * m_unitScale;  // Convert mm/min to m/s
            addSegment();
            break;
            
        case 2:  // CW arc (simplified to linear for now)
        case 3:  // CCW arc
            m_currentState.motionMode = gCode;
            m_pendingTarget = Vec3(x, y, z) * m_unitScale;
            m_pendingFeed = f / 60.0 * m_unitScale;
            addSegment();  // TODO: Proper arc interpolation
            break;
            
        case 17: // XY plane
        case 18: // XZ plane  
        case 19: // YZ plane
            // Plane selection (mostly relevant for arcs)
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
            // Mark end but don't do anything special
            break;
            
        default:
            break;
    }
}

void GCodeInterpreter::addSegment() {
    Vec3 startPos = m_currentState.position;
    Vec3 endPos = m_pendingTarget;
    
    if (!m_currentState.absoluteMode) {
        // Incremental mode
        endPos = startPos + m_pendingTarget;
    }
    
    // Calculate distance and duration
    Vec3 delta = endPos - startPos;
    double distance = delta.length();
    
    if (distance < 1e-12) {
        // No actual movement
        return;
    }
    
    double feedRate = m_pendingFeed;
    if (feedRate < 1e-12) {
        feedRate = 0.001;  // Minimum feed to avoid infinite duration
    }
    
    double duration = distance / feedRate;
    
    TrajectorySegment seg;
    seg.startTime = m_totalDuration;
    seg.endTime = m_totalDuration + duration;
    seg.startPos = startPos;
    seg.endPos = endPos;
    seg.feedRate = feedRate;
    seg.spindleRPM = m_currentState.spindleRPM;
    seg.motionType = m_currentState.motionMode;
    
    m_segments.push_back(seg);
    
    // Update state
    m_totalDuration = seg.endTime;
    m_currentState.position = endPos;
    m_currentState.feedRate = feedRate;
    m_currentState.time = m_totalDuration;
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
        return state;
    }
    
    if (simTime >= m_totalDuration) {
        state.position = m_segments.back().endPos;
        state.feedRate = m_segments.back().feedRate;
        state.spindleRPM = m_segments.back().spindleRPM;
        state.time = m_totalDuration;
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
    
    // Linear interpolation within segment
    double t = (simTime - seg.startTime) / (seg.endTime - seg.startTime);
    t = std::max(0.0, std::min(1.0, t));
    
    state.position = seg.startPos + (seg.endPos - seg.startPos) * t;
    state.feedRate = seg.feedRate;
    state.spindleRPM = seg.spindleRPM;
    state.time = simTime;
    state.spindleOn = seg.spindleRPM > 0;
    
    // Calculate direction of motion
    Vec3 delta = seg.endPos - seg.startPos;
    if (delta.lengthSq() > 1e-12) {
        state.direction = delta.normalized();
    }
    
    return state;
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
        minCorner.x = std::min({minCorner.x, seg.startPos.x, seg.endPos.x});
        minCorner.y = std::min({minCorner.y, seg.startPos.y, seg.endPos.y});
        minCorner.z = std::min({minCorner.z, seg.startPos.z, seg.endPos.z});
        maxCorner.x = std::max({maxCorner.x, seg.startPos.x, seg.endPos.x});
        maxCorner.y = std::max({maxCorner.y, seg.startPos.y, seg.endPos.y});
        maxCorner.z = std::max({maxCorner.z, seg.startPos.z, seg.endPos.z});
    }
}

} // namespace edgepredict
