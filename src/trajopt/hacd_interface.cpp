#include "hacd_interface.hpp"
#include <hacdHACD.h>
using HACD::Vec3;
using HACD::Real;
using namespace std;

void polygonMeshFromPointsTriangles(OpenRAVE::TriMesh& mesh, const vector< Vec3<Real> >& points, const vector< Vec3<long> >& triangles) {
  mesh.vertices.resize(points.size());
  for (int i=0; i < mesh.vertices.size(); ++i) {
    mesh.vertices[i].x = points[i].X();
    mesh.vertices[i].y = points[i].Y();
    mesh.vertices[i].z = points[i].Z();
  }
  mesh.indices.resize(triangles.size()*3);
  for (int i=0; i < mesh.indices.size(); i += 3) {
    mesh.indices[i] = triangles[i/3].X();
    mesh.indices[i+1] = triangles[i/3].Y();
    mesh.indices[i+2] = triangles[i/3].Z();
  }
}
void pointsTrianglesFromPolygonMesh(OpenRAVE::TriMesh& mesh, vector< Vec3<Real> >& points, vector< Vec3<long> >& triangles) {
  points.resize(mesh.vertices.size());
  for (int i=0; i < points.size(); ++i) {
    points[i].X() = mesh.vertices[i].x;
    points[i].Y() = mesh.vertices[i].y;
    points[i].Z() = mesh.vertices[i].z;
  }
  triangles.resize(mesh.indices.size()/3);
  for (size_t i = 0; i < triangles.size(); ++i) {
    triangles[i].X() = mesh.indices[3*i];
    triangles[i].Y() = mesh.indices[3*i+1];
    triangles[i].Z() = mesh.indices[3*i+2];
  }
}

void CallBack(const char * msg, double progress, double concavity, size_t nVertices)
{
    std::cout << msg;
}

vector<OpenRAVE::TriMesh> ConvexDecompHACD(OpenRAVE::TriMesh mesh, float concavity) {
  int minClusters = 2;
  bool addExtraDistPoints = true;
  bool addFacesPoints = true;
  float ccConnectDist = 30;
  int targetNTrianglesDecimatedMesh = 3000;

  vector< Vec3<Real> > points;
  vector< Vec3<long> > triangles;
  pointsTrianglesFromPolygonMesh(mesh, points, triangles);

  HACD::HeapManager * heapManager = HACD::createHeapManager(65536*(1000));

  HACD::HACD * const myHACD = HACD::CreateHACD(heapManager);
  myHACD->SetPoints(&points[0]);
  myHACD->SetNPoints(points.size());
  myHACD->SetTriangles(&triangles[0]);
  myHACD->SetNTriangles(triangles.size());
  myHACD->SetCompacityWeight(0.0001);
  myHACD->SetVolumeWeight(0.0);
  myHACD->SetConnectDist(ccConnectDist);               // if two connected components are seperated by distance < ccConnectDist
                            // then create a virtual edge between them so the can be merged during
                            // the simplification process

  myHACD->SetNClusters(minClusters);                     // minimum number of clusters
  myHACD->SetNVerticesPerCH(100);                      // max of 100 vertices per convex-hull
  myHACD->SetConcavity(concavity);                     // maximum concavity
  myHACD->SetSmallClusterThreshold(0.25);              // threshold to detect small clusters
  myHACD->SetNTargetTrianglesDecimatedMesh(targetNTrianglesDecimatedMesh); // # triangles in the decimated mesh
  myHACD->SetCallBack(&CallBack);
  myHACD->SetAddExtraDistPoints(addExtraDistPoints);
  myHACD->SetAddFacesPoints(addFacesPoints);

  myHACD->Compute();
  int nClusters = myHACD->GetNClusters();
  vector<OpenRAVE::TriMesh> outmeshes(nClusters);
  for (int i=0; i < nClusters; ++i) {
    size_t nPoints = myHACD->GetNPointsCH(i);
    size_t nTriangles = myHACD->GetNTrianglesCH(i);
    vector < Vec3<Real> > hullpoints(nPoints);
    vector < Vec3<long> > hulltriangles(nTriangles);
    myHACD->GetCH(i, &hullpoints[0], &hulltriangles[0]);
    polygonMeshFromPointsTriangles(outmeshes[i], hullpoints, hulltriangles);
  }

  HACD::DestroyHACD(myHACD);
  HACD::releaseHeapManager(heapManager);

  return outmeshes;
}
