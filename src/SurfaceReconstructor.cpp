/**
 * @file SurfaceReconstructor.cpp
 * @brief Marching Cubes surface reconstruction from SPH particle cloud
 */

#include "SurfaceReconstructor.h"
#include "CoordinateSystem.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace edgepredict {

// ============================================================================
// Marching Cubes Lookup Tables (standard 256-entry)
// ============================================================================

const int SurfaceReconstructor::edgeTable[256] = {
    0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0
};

const int SurfaceReconstructor::triTable[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

// ============================================================================
// Density Field Construction
// ============================================================================

std::vector<double> SurfaceReconstructor::buildDensityField(
    const std::vector<SPHParticle>& particles,
    const Vec3& gridMin, int nx, int ny, int nz,
    double cellSize, double radius) {

    std::vector<double> field(static_cast<size_t>(nx) * ny * nz, 0.0);
    double r2 = radius * radius;
    int cellsPerRadius = static_cast<int>(std::ceil(radius / cellSize)) + 1;

    for (const auto& p : particles) {
        if (p.status == ParticleStatus::INACTIVE) continue;

        int ci = static_cast<int>((p.x - gridMin.x) / cellSize);
        int cj = static_cast<int>((p.y - gridMin.y) / cellSize);
        int ck = static_cast<int>((p.z - gridMin.z) / cellSize);

        int imin = std::max(0, ci - cellsPerRadius);
        int imax = std::min(nx - 1, ci + cellsPerRadius);
        int jmin = std::max(0, cj - cellsPerRadius);
        int jmax = std::min(ny - 1, cj + cellsPerRadius);
        int kmin = std::max(0, ck - cellsPerRadius);
        int kmax = std::min(nz - 1, ck + cellsPerRadius);

        for (int iz = kmin; iz <= kmax; ++iz) {
            for (int iy = jmin; iy <= jmax; ++iy) {
                for (int ix = imin; ix <= imax; ++ix) {
                    double gx = gridMin.x + ix * cellSize;
                    double gy = gridMin.y + iy * cellSize;
                    double gz = gridMin.z + iz * cellSize;
                    double dx = gx - p.x, dy = gy - p.y, dz = gz - p.z;
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < r2) {
                        double q = std::sqrt(d2) / radius;
                        // Cubic spline kernel (normalized)
                        double w = 0;
                        if (q < 0.5) w = 6.0*(q*q*q - q*q) + 1.0;
                        else if (q < 1.0) w = 2.0*(1.0-q)*(1.0-q)*(1.0-q);
                        size_t idx = static_cast<size_t>(iz)*ny*nx + static_cast<size_t>(iy)*nx + ix;
                        field[idx] += w;
                    }
                }
            }
        }
    }
    return field;
}

// ============================================================================
// Marching Cubes
// ============================================================================

static Vec3 vertexInterp(double iso, const Vec3& p1, const Vec3& p2, double v1, double v2) {
    if (std::abs(iso - v1) < 1e-12) return p1;
    if (std::abs(iso - v2) < 1e-12) return p2;
    if (std::abs(v1 - v2) < 1e-12) return p1;
    double mu = (iso - v1) / (v2 - v1);
    return Vec3(p1.x + mu*(p2.x-p1.x), p1.y + mu*(p2.y-p1.y), p1.z + mu*(p2.z-p1.z));
}

Mesh SurfaceReconstructor::marchingCubes(const std::vector<double>& field,
                                          const Vec3& gridMin,
                                          int nx, int ny, int nz,
                                          double cellSize, double isoValue) {
    Mesh mesh;
    auto idx = [&](int x, int y, int z) -> size_t {
        return static_cast<size_t>(z)*ny*nx + static_cast<size_t>(y)*nx + x;
    };

    // Vertex deduplication
    std::unordered_map<int64_t, int> vertexMap;
    auto getOrAddVertex = [&](const Vec3& pos) -> int {
        // Quantize to cellSize/100 for dedup
        int64_t qx = static_cast<int64_t>(pos.x / (cellSize*0.01));
        int64_t qy = static_cast<int64_t>(pos.y / (cellSize*0.01));
        int64_t qz = static_cast<int64_t>(pos.z / (cellSize*0.01));
        int64_t key = qx * 1000000007LL + qy * 1000003LL + qz;
        auto it = vertexMap.find(key);
        if (it != vertexMap.end()) return it->second;
        int id = static_cast<int>(mesh.nodes.size());
        FEMNode node;
        node.position = pos;
        node.originalPosition = pos;
        mesh.nodes.push_back(node);
        vertexMap[key] = id;
        return id;
    };

    for (int z = 0; z < nz-1; ++z) {
        for (int y = 0; y < ny-1; ++y) {
            for (int x = 0; x < nx-1; ++x) {
                double val[8];
                val[0] = field[idx(x,   y,   z)];
                val[1] = field[idx(x+1, y,   z)];
                val[2] = field[idx(x+1, y+1, z)];
                val[3] = field[idx(x,   y+1, z)];
                val[4] = field[idx(x,   y,   z+1)];
                val[5] = field[idx(x+1, y,   z+1)];
                val[6] = field[idx(x+1, y+1, z+1)];
                val[7] = field[idx(x,   y+1, z+1)];

                int cubeIndex = 0;
                if (val[0] < isoValue) cubeIndex |= 1;
                if (val[1] < isoValue) cubeIndex |= 2;
                if (val[2] < isoValue) cubeIndex |= 4;
                if (val[3] < isoValue) cubeIndex |= 8;
                if (val[4] < isoValue) cubeIndex |= 16;
                if (val[5] < isoValue) cubeIndex |= 32;
                if (val[6] < isoValue) cubeIndex |= 64;
                if (val[7] < isoValue) cubeIndex |= 128;

                if (edgeTable[cubeIndex] == 0) continue;

                Vec3 corners[8];
                corners[0] = Vec3(gridMin.x+(x)*cellSize,   gridMin.y+(y)*cellSize,   gridMin.z+(z)*cellSize);
                corners[1] = Vec3(gridMin.x+(x+1)*cellSize, gridMin.y+(y)*cellSize,   gridMin.z+(z)*cellSize);
                corners[2] = Vec3(gridMin.x+(x+1)*cellSize, gridMin.y+(y+1)*cellSize, gridMin.z+(z)*cellSize);
                corners[3] = Vec3(gridMin.x+(x)*cellSize,   gridMin.y+(y+1)*cellSize, gridMin.z+(z)*cellSize);
                corners[4] = Vec3(gridMin.x+(x)*cellSize,   gridMin.y+(y)*cellSize,   gridMin.z+(z+1)*cellSize);
                corners[5] = Vec3(gridMin.x+(x+1)*cellSize, gridMin.y+(y)*cellSize,   gridMin.z+(z+1)*cellSize);
                corners[6] = Vec3(gridMin.x+(x+1)*cellSize, gridMin.y+(y+1)*cellSize, gridMin.z+(z+1)*cellSize);
                corners[7] = Vec3(gridMin.x+(x)*cellSize,   gridMin.y+(y+1)*cellSize, gridMin.z+(z+1)*cellSize);

                Vec3 vertList[12];
                if (edgeTable[cubeIndex] & 1)    vertList[0]  = vertexInterp(isoValue, corners[0], corners[1], val[0], val[1]);
                if (edgeTable[cubeIndex] & 2)    vertList[1]  = vertexInterp(isoValue, corners[1], corners[2], val[1], val[2]);
                if (edgeTable[cubeIndex] & 4)    vertList[2]  = vertexInterp(isoValue, corners[2], corners[3], val[2], val[3]);
                if (edgeTable[cubeIndex] & 8)    vertList[3]  = vertexInterp(isoValue, corners[3], corners[0], val[3], val[0]);
                if (edgeTable[cubeIndex] & 16)   vertList[4]  = vertexInterp(isoValue, corners[4], corners[5], val[4], val[5]);
                if (edgeTable[cubeIndex] & 32)   vertList[5]  = vertexInterp(isoValue, corners[5], corners[6], val[5], val[6]);
                if (edgeTable[cubeIndex] & 64)   vertList[6]  = vertexInterp(isoValue, corners[6], corners[7], val[6], val[7]);
                if (edgeTable[cubeIndex] & 128)  vertList[7]  = vertexInterp(isoValue, corners[7], corners[4], val[7], val[4]);
                if (edgeTable[cubeIndex] & 256)  vertList[8]  = vertexInterp(isoValue, corners[0], corners[4], val[0], val[4]);
                if (edgeTable[cubeIndex] & 512)  vertList[9]  = vertexInterp(isoValue, corners[1], corners[5], val[1], val[5]);
                if (edgeTable[cubeIndex] & 1024) vertList[10] = vertexInterp(isoValue, corners[2], corners[6], val[2], val[6]);
                if (edgeTable[cubeIndex] & 2048) vertList[11] = vertexInterp(isoValue, corners[3], corners[7], val[3], val[7]);

                for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                    int a = getOrAddVertex(vertList[triTable[cubeIndex][i]]);
                    int b = getOrAddVertex(vertList[triTable[cubeIndex][i+1]]);
                    int c = getOrAddVertex(vertList[triTable[cubeIndex][i+2]]);
                    Triangle tri;
                    tri.indices[0] = a;
                    tri.indices[1] = b;
                    tri.indices[2] = c;
                    Vec3 e1 = mesh.nodes[b].position - mesh.nodes[a].position;
                    Vec3 e2 = mesh.nodes[c].position - mesh.nodes[a].position;
                    tri.normal = e1.cross(e2).normalized();
                    mesh.triangles.push_back(tri);
                }
            }
        }
    }
    return mesh;
}

// ============================================================================
// Laplacian Smoothing
// ============================================================================

void SurfaceReconstructor::smoothMesh(Mesh& mesh, int passes) {
    if (mesh.nodes.empty() || passes <= 0) return;

    // Build adjacency
    std::vector<std::vector<int>> adj(mesh.nodes.size());
    for (const auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; ++i) {
            int a = tri.indices[i];
            int b = tri.indices[(i+1)%3];
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<Vec3> newPos(mesh.nodes.size());
        for (size_t i = 0; i < mesh.nodes.size(); ++i) {
            if (adj[i].empty()) {
                newPos[i] = mesh.nodes[i].position;
                continue;
            }
            Vec3 avg = Vec3::zero();
            for (int nb : adj[i]) avg += mesh.nodes[nb].position;
            avg = avg / static_cast<double>(adj[i].size());
            // Blend: 50% original + 50% average neighbor
            newPos[i] = mesh.nodes[i].position * 0.5 + avg * 0.5;
        }
        for (size_t i = 0; i < mesh.nodes.size(); ++i) {
            mesh.nodes[i].position = newPos[i];
        }
    }

    // Recompute normals
    for (auto& tri : mesh.triangles) {
        const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
        const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
        const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
        tri.normal = (v1 - v0).cross(v2 - v0).normalized();
    }
}

// ============================================================================
// Scalar Interpolation
// ============================================================================

void SurfaceReconstructor::interpolateScalars(Mesh& mesh,
                                               const std::vector<SPHParticle>& particles,
                                               double radius) {
    double r2 = radius * radius;
    for (auto& node : mesh.nodes) {
        double totalW = 0;
        double tempSum = 0;
        double stressSum = 0;
        for (const auto& p : particles) {
            if (p.status == ParticleStatus::INACTIVE) continue;
            double dx = node.position.x - p.x;
            double dy = node.position.y - p.y;
            double dz = node.position.z - p.z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < r2) {
                double w = 1.0 - std::sqrt(d2) / radius;
                w = w * w; // quadratic falloff
                tempSum += p.temperature * w;
                double vm = std::sqrt(p.stress_xx*p.stress_xx + p.stress_yy*p.stress_yy + p.stress_zz*p.stress_zz);
                stressSum += vm * w;
                totalW += w;
            }
        }
        if (totalW > 1e-15) {
            node.temperature = tempSum / totalW;
            node.stress = stressSum / totalW;
        }
    }
}

// ============================================================================
// Main Reconstruct Entry Point
// ============================================================================

Mesh SurfaceReconstructor::reconstruct(const std::vector<SPHParticle>& particles,
                                        const ReconstructionParams& params) {
    // Count active particles
    int activeCount = 0;
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) activeCount++;
    }

    if (activeCount == 0) {
        std::cerr << "[SurfaceReconstructor] No active particles — returning empty mesh" << std::endl;
        return Mesh();
    }

    // Compute TIGHT bounding box from actual particle positions (not domain padding)
    BoundingBox bb;
    for (const auto& p : particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            bb.expand(Vec3(p.x, p.y, p.z));
        }
    }

    // Use 2× smoothing radius as margin so the isosurface stays within bounds
    double pad = params.smoothingRadius * 2.0;
    Vec3 gridMin(bb.min.x - pad, bb.min.y - pad, bb.min.z - pad);
    Vec3 gridMax(bb.max.x + pad, bb.max.y + pad, bb.max.z + pad);

    // Determine cell size for reconstruction. Use params.cellSize as target,
    // but enforce it is at least smoothingRadius/2 (won't over-refine) and at
    // most 2*smoothingRadius (won't under-resolve).
    double cellSize = params.cellSize;
    cellSize = std::max(cellSize, params.smoothingRadius * 0.5);
    cellSize = std::min(cellSize, params.smoothingRadius * 2.0);

    int nx = static_cast<int>((gridMax.x - gridMin.x) / cellSize) + 2;
    int ny = static_cast<int>((gridMax.y - gridMin.y) / cellSize) + 2;
    int nz = static_cast<int>((gridMax.z - gridMin.z) / cellSize) + 2;

    // Cap grid to 128³ (< 8 MB at double precision) — if we hit this, enlarge cellSize
    const int maxDim = 128;
    if (nx > maxDim || ny > maxDim || nz > maxDim) {
        // Scale cell size up so the domain fits inside maxDim³
        double scaleX = (nx > maxDim) ? static_cast<double>(nx) / maxDim : 1.0;
        double scaleY = (ny > maxDim) ? static_cast<double>(ny) / maxDim : 1.0;
        double scaleZ = (nz > maxDim) ? static_cast<double>(nz) / maxDim : 1.0;
        double scale  = std::max({scaleX, scaleY, scaleZ});
        cellSize *= scale;          // make cells larger so grid shrinks
        nx = std::min(nx, maxDim);
        ny = std::min(ny, maxDim);
        nz = std::min(nz, maxDim);
        std::cout << "[SurfaceReconstructor] Grid capped to " << nx << "x" << ny << "x" << nz
                  << " (cell size adjusted to " << cellSize * 1e6 << " μm)" << std::endl;
    }

    std::cout << "[SurfaceReconstructor] Grid: " << nx << "x" << ny << "x" << nz
              << " = " << (static_cast<long long>(nx)*ny*nz) << " voxels"
              << " (" << activeCount << " active particles)"
              << " cellSize=" << cellSize*1e6 << " μm" << std::endl;

    // Step 1: Build density field (use corrected cellSize throughout)
    auto field = buildDensityField(particles, gridMin, nx, ny, nz,
                                    cellSize, params.smoothingRadius);

    // Step 2: Marching Cubes (use corrected cellSize)
    Mesh mesh = marchingCubes(field, gridMin, nx, ny, nz,
                               cellSize, params.isoValue);

    std::cout << "[SurfaceReconstructor] Extracted " << mesh.nodes.size()
              << " vertices, " << mesh.triangles.size() << " triangles" << std::endl;

    // Step 3: Laplacian smoothing
    if (params.smoothingPasses > 0) {
        smoothMesh(mesh, params.smoothingPasses);
    }

    // Step 4: Interpolate scalar fields
    if (params.interpolateScalars) {
        interpolateScalars(mesh, particles, params.smoothingRadius * 2.0);
    }

    return mesh;
}

} // namespace edgepredict
