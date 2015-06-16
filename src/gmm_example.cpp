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

  classifier.accumulate_points(tester/1);
  classifier.accumulate_points(tester/1);
  classifier.accumulate_points(tester/1);
  classifier.accumulate_points(tester/1);
  classifier.accumulate_points(tester/1);

  ResultGmm gmm_result;
  gmm_result = classifier.pop_gmm_results();

  std::cout<<"Gmm Evaluation: "<<value<<std::endl;
  std::cout<<"Gmm mode: "<<gmm_result.winner_mode<<std::endl;
  std::cout<<"Gmm candidates :"<<gmm_result.NormalizedProb<<std::endl;
	return 0;
}
