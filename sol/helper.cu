#include <math.h>
#include <fstream>

#include "helper.h"

using namespace std;

// geoDistance computes geographical distance (lat1, lat1) and (lat2, lon2)
__device__ float geoDistance(float lat1, float lon1, float lat2, float lon2)
{
    float phi1 = (90.f - lat1) * DEGREE_TO_RADIANS;
    float phi2 = (90.f - lat2) * DEGREE_TO_RADIANS;

    float theta1 = lon1 * DEGREE_TO_RADIANS;
    float theta2 = lon2 * DEGREE_TO_RADIANS;

    float cs = sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2);
    if (cs > 1) {
        cs = 1;
    } else if (cs < -1) {
        cs = -1;
    }

    return 6371.f * acos(cs);
}
