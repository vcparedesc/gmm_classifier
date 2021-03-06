#include "gmm_classifier.hpp"

gmm_classifier::gmm_classifier()
{
  Models = new Modelp[4];
  Probabilities = VectorXd::Zero(4);

	// Load Parameters using meta template
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Standing.yaml",Models[0]);
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Stand2Walk.yaml",Models[1]);
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Walk2Stand.yaml",Models[2]);
  BEHAVIOR<8>::Load("package://gmm_classifier/Models/Walking.yaml",Models[3]);

  nBehaviors = 4;
  nFeatures = Models[0].mu.rows();
  Probabilities = VectorXd::Zero(nBehaviors);


}

gmm_classifier::~gmm_classifier()
{
  //delete Models;
}

double gmm_classifier::evalGmm(Behaviors::MODE mode, VectorXd features_vector)
{
  double result = 0;

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

  if(Models[mode].nClusters != 1)
    {
      vec = (features_vector.transpose() - Models[mode].mu.block(n_cluster,0,1,8)).transpose();
    }else{
      vec = (features_vector.transpose() - Models[mode].mu.transpose()).transpose();
    }


  result = 1 / sqrt( pow(2 * 3.141516,8) * Models[mode].sigma_det[n_cluster]) * exp((double)(-0.5 * vec.transpose() * Models[mode].sigma_inv[n_cluster] * vec));

  return result;
}

void gmm_classifier::accumulate_points(VectorXd features_vector)
{
  for(int i = 0; i < nBehaviors; i++)
    {
      // Using log(1+value) to reduce values.
      Probabilities(i) = Probabilities(i) + log(evalGmm((Behaviors::MODE)i,features_vector) + 1);
    }
}

ResultGmm gmm_classifier::pop_gmm_results()
{
  ResultGmm result;
  result.NormalizedProb = VectorXd::Zero(nBehaviors);
  int Index = 0;
  double temp;


  temp = Probabilities(0);
  for (int i = 0; i < nBehaviors; i++)
    {
      if(temp > Probabilities(i))
        {
          temp = Probabilities(i);
          Index = i;
        }
    }
  result.winner_mode = (Behaviors::MODE)Index;
  if(Probabilities != VectorXd::Zero(nBehaviors))
    {
      Probabilities.normalize();
    }

  result.NormalizedProb = Probabilities;

  // Reset Values
  Probabilities = VectorXd::Zero(nBehaviors);
  current_mode = result.winner_mode;
  currentBehavior = Index;

  return result;
}

void gmm_classifier::reset_probabilities()
{
  Probabilities = VectorXd::Zero(nBehaviors);
}
