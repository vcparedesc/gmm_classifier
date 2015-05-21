#include "gmm_classifier.hpp"

gmm_classifier::gmm_classifier()
{
  Models = new Modelp[3];

	// Load Parameters using meta template
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Standing.yaml",Models[0]);
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Stand2Walk.yaml",Models[1]);
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Walk2Stand.yaml",Models[2]);
}

gmm_classifier::~gmm_classifier()
{
  //delete Models;
}

double gmm_classifier::evalGmm(Behaviors::MODE mode, VectorXd features_vector)
{
  double result = 0;
  std::cout<<"Entering evalGmm"<<std::endl;
  for (int i = 0; i < Models[mode].nClusters; i ++)
  {
    result += Models[mode].phi[i] * NDgaussian(mode, i, features_vector);
  }

  return result;
}

double gmm_classifier::NDgaussian(Behaviors::MODE mode, int n_cluster, VectorXd features_vector)
{
  double result;
  VectorXd vec;

  vec = (features_vector.transpose() - Models[mode].mu.block(n_cluster,0,1,8)).transpose();
  std::cout<<"Test Mean diff"<<vec<<std::endl;
  result = 1 / sqrt( pow(2 * 3.141516,8) * Models[mode].sigma_det[n_cluster]) * exp(-0.5 * vec.transpose() * Models[mode].sigma_inv[n_cluster] * vec);

  return result;
}