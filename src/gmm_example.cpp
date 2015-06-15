#include <iostream>
#include "gmm_classifier.hpp"
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

int main(int argc, char **argv)
{
	gmm_classifier classifier;
  double value;
  VectorXd tester = VectorXd::Zero(8);
  tester << -0.00762,  -0.0037391, 0.0417344, -0.1975026, 0.0001198, 0.0003595, -0.0001616, -0.0010937;
  std::cout<<"mu"<<classifier.Models[0].sigma[0]<<std::endl;
  value = classifier.evalGmm(Behaviors::STANDING, tester);

  std::cout<<"Gmm Evaluation: "<<value<<std::endl;
	return 0;
}
