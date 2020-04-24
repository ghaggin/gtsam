/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SFMExample_SmartFactor.cpp
 * @brief   A structure-from-motion problem on a simulated dataset, using smart
 * projection factor
 * @author  ghaggin
 */

// In GTSAM, measurement functions are represented as 'factors'.
// The factor we used here is SmartProjectionPoseFactor.
// Every smart factor represent a single landmark, seen from multiple cameras.
// The SmartProjectionPoseFactor only optimizes for the poses of a camera,
// not the calibration, which is assumed known.
#include <gtsam/slam/SmartProjectionPoseFactor.h>

// Camera calibration
#include <gtsam/geometry/Cal3Fisheye.h>

// For an explanation of these headers, see SFMExample.cpp
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include "SFMdata.h"

// For disturbing the intial estimates and measurements
#include <random>

using namespace std;
using namespace gtsam;

// Make the typename short so it looks much cleaner
typedef SmartProjectionPoseFactor<Cal3Fisheye> SmartFactor;
typedef PinholePose<Cal3Fisheye> Camera;

/* ************************************************************************* */
int main(int argc, char* argv[]) {
  // Define the camera calibration parameters
  boost::shared_ptr<Cal3Fisheye> calibration(new Cal3Fisheye(
      500, 100, 0.1, 320, 240, 1e-2, 2.0 * 1e-2, 3.0 * 1e-2, 4.0 * 1e-2));

  // Define the camera observation noise model
  noiseModel::Isotropic::shared_ptr meas_noise =
      noiseModel::Isotropic::Sigma(2, 1.0);  // one pixel in u and v

  // Create the set of ground-truth landmarks and poses
  // from SFMdata
  vector<Point3> points = createPoints();
  vector<Pose3> poses = createPoses();

  // Create a factor graph
  NonlinearFactorGraph graph;

  // For each landmark, simulate measurement from each camera pose
  // adding it to the smart factor, then add the smart factor to the graph
  for (size_t j = 0; j < points.size(); ++j) {
    // Create smart factor for this landmark
    auto smartfactor = boost::make_shared<SmartFactor>(meas_noise, calibration);

    for (size_t i = 0; i < poses.size(); ++i) {
      // generate measurement in image plane
      Camera camera(poses[i], calibration);
      Point2 measurement = camera.project(points[j]);

      // add measurement to the smart factor for camera
      // frame i
      smartfactor->add(measurement, i);
    }

    // insert the smart factor in the graph
    graph.add(smartfactor);
  }

  // Add a prior on pose x0. This indirectly specifies where the origin is.
  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
  noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3)).finished());
  graph.emplace_shared<PriorFactor<Pose3> >(0, poses[0], noise);

  // Because the structure-from-motion problem has a scale ambiguity, the
  // problem is still under-constrained. Here we add a prior on the second pose
  // x1, so this will fix the scale by indicating the distance between x0 and
  // x1. Because these two are fixed, the rest of the poses will be also be
  // fixed.
  graph.emplace_shared<PriorFactor<Pose3> >(1, poses[1],
                                            noise);  // add directly to graph

  graph.print("Factor Graph:\n");

  // Create the initial estimate to the solution
  // Intentionally initialize the variables off from the ground truth
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> rnd_angle(0, 0.01);
  std::normal_distribution<> rnd_pos(0, 0.5);

  Values initialEstimate;
  for (size_t i = 0; i < poses.size(); ++i) {
    Pose3 delta(Rot3::Rodrigues(rnd_angle(gen), rnd_angle(gen), rnd_angle(gen)),
                Point3(rnd_pos(gen), rnd_pos(gen), rnd_pos(gen)));
    initialEstimate.insert(i, poses[i].compose(delta));
  }
  initialEstimate.print("Initial Estimates:\n");

  // Optimize the graph and print results
  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
  Values result = optimizer.optimize();
  result.print("Final results:\n");

  // A smart factor represent the 3D point inside the factor, not as a variable.
  // The 3D position of the landmark is not explicitly calculated by the
  // optimizer. To obtain the landmark's 3D position, we use the "point" method
  // of the smart factor.
  Values landmark_result;
  for (size_t j = 0; j < points.size(); ++j) {
    // The graph stores Factor shared_ptrs, so we cast back to a SmartFactor
    // first
    SmartFactor::shared_ptr smart =
        boost::dynamic_pointer_cast<SmartFactor>(graph[j]);
    if (smart) {
      // The output of point() is in boost::optional<Point3>, as sometimes
      // the triangulation operation inside smart factor will encounter
      // degeneracy.
      boost::optional<Point3> point = smart->point(result);
      if (point)  // ignore if boost::optional return nullptr
        landmark_result.insert(j, *point);
    }
  }

  landmark_result.print("Landmark results:\n");
  cout << "final error: " << graph.error(result) << endl;
  cout << "number of iterations: " << optimizer.iterations() << endl;

  return 0;
}
/* ************************************************************************* */
