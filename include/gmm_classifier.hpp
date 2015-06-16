#ifndef GAUSSIAN_MIXTURE_HPP
#define GAUSSIAN_MIXTURE_HPP

#include <iostream>
#include <stdio.h>
#include <yaml_eigen_utilities/yaml_eigen_utilities.hpp>
#include <roslib_utilities/ros_package_utilities.hpp>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <string>

using namespace std;
using namespace YAML;
using boost::filesystem::path;
using namespace yaml_utilities;
using namespace Eigen;
using namespace common;

namespace Behaviors{
  enum MODE{
    STANDING = 0,
    STAND2WALK = 1,
    WALK2STAND = 2
  };
}

struct Modelp
{
  MatrixXd mu;
  MatrixXd *sigma;
  MatrixXd *sigma_inv;
  double *sigma_det;
  VectorXd phi;
  int nClusters;
};

struct ResultGmm
{
  Behaviors::MODE winner_mode;
  VectorXd NormalizedProb;
};

template< int features > 
class BEHAVIOR{
	public:
  static inline void Load(string address, Modelp &behavior)
  {
  Node bCluster;
  double scalar;
  string filepath = roslib_utilities::resolve_local_url(address.c_str()).string();
  yaml_utilities::yaml_read_file(filepath, bCluster);

  const Node &nclusters = bCluster["nClusters"];

  nclusters >> scalar;

  const Node &mu = bCluster["mu"];
  behavior.nClusters = scalar;
  behavior.sigma = new MatrixXd[(int)scalar];
  behavior.sigma_inv = new MatrixXd[(int)scalar];
  behavior.sigma_det = new double[(int)scalar];
  behavior.mu.resize((int) scalar, features);
  behavior.phi.resize((int) scalar);

  for(int i = 1; i <= scalar; i++)
  {
    string temp = "sigma";
    char *cstr = &temp[0u];
    string s = std::to_string(i);
    strcat(cstr, s.c_str());

    const Node &sigma = bCluster[cstr];
    behavior.sigma[i-1].resize(features,features);
    behavior.sigma_inv[i-1].resize(features,features);
    sigma >> behavior.sigma[i-1];
    behavior.sigma_inv[i-1] = behavior.sigma[i-1].lu().inverse();
    behavior.sigma_det[i-1] = behavior.sigma[i-1].lu().determinant();
  }

  const Node &phi = bCluster["phi"]; 

  mu >> behavior.mu;
  phi >> behavior.phi;

  }		
};

class gmm_classifier{
private:
  int nFeatures;
  int nBehaviors;
  int currentBehavior;
  double NDgaussian(Behaviors::MODE mode, int n_cluster, VectorXd features_vector);

  Behaviors::MODE current_mode;

public:
  Modelp* Models;
  gmm_classifier();
  ~gmm_classifier();
  double evalGmm(Behaviors::MODE mode, VectorXd features_vector);
  void accumulate_points(VectorXd features_vector);
  ResultGmm pop_gmm_results();
  void reset_probabilities();

  VectorXd Probabilities;

};

#endif
