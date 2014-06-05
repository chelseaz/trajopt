#pragma once
#include <openrave-core.h>
#include <vector>
#include "macros.h"

TRAJOPT_API std::vector<OpenRAVE::TriMesh> ConvexDecompHACD(OpenRAVE::TriMesh mesh, float concavity);
