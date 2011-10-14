/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation, 
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file visualSLAM.cpp
 * @date Jan 14, 2010
 * @author richard
 */

#include <gtsam/slam/visualSLAM.h>
#include <gtsam/nonlinear/TupleValues-inl.h>
#include <gtsam/nonlinear/NonlinearOptimizer-inl.h>
#include <gtsam/nonlinear/NonlinearFactorGraph-inl.h>

namespace gtsam {
	INSTANTIATE_TUPLE_VALUES2(visualSLAM::PoseValues, visualSLAM::PointValues)
	INSTANTIATE_NONLINEAR_FACTOR_GRAPH(visualSLAM::Values)
	INSTANTIATE_NONLINEAR_OPTIMIZER(visualSLAM::Graph, visualSLAM::Values)

}
