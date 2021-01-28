/**
 * particle_filter.cpp
 *
 * Created on: Jan 28, 2021
 * Author: Ruixin Liu
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>
#include <cfloat>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;
using std::unordered_set;

static double yaw_threshold = 0.1;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  this->num_particles = 10;  // TODO: Set the number of particles
  is_initialized = true;
  
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta,std[2]);
  
  for(int i = 0; i < this->num_particles; i++)
    // Assign values one by one
  {
    Particle newP;
    newP.id = i;
    newP.x = dist_x(gen);
    newP.y = dist_y(gen);
    newP.theta = dist_theta(gen);
    newP.weight = 1.0;
    this->particles.push_back(newP);
    this->weights.push_back(newP.weight);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  normal_distribution<double> dist_x(0.0,std_pos[0]);
  normal_distribution<double> dist_y(0.0,std_pos[1]);
  normal_distribution<double> dist_theta(0.0,std_pos[2]);
  
  for(auto& p : particles)
  {
    if(std::fabs(yaw_rate) > yaw_threshold)
    {
      p.x = p.x+(velocity/yaw_rate)*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta)) + dist_x(gen);
      p.y = p.y+(velocity/yaw_rate)*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t)) + dist_y(gen);
      p.theta = p.theta+yaw_rate*delta_t + dist_theta(gen);
    }
    else
    {
      p.x = p.x + velocity * delta_t * cos(p.theta) + dist_x(gen);
      p.y = p.y + velocity * delta_t * sin(p.theta) + dist_y(gen);
      p.theta = p.theta + dist_theta(gen);
    }
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // The data association part is merged into updateWeights() for simplicity.
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(int i = 0; i < this->particles.size(); i++)
  {
    auto& p = particles[i];
    p.weight = 1.0;
	vector<LandmarkObs> map_observations;// Observations will be in map frame according to this particle's position
    for(int i = 0; i < observations.size(); i++)
    {
      LandmarkObs map_obs;
      
      // transform coordinates
      map_obs.x = observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta) + p.x;
      map_obs.y = observations[i].x * sin(p.theta) + observations[i].y * cos(p.theta) + p.y;
      
      // find all the lankmarks that are within sensor range
      unordered_set<int> available_landmarks_idx;
      const auto& total_landmarks = map_landmarks.landmark_list;
      for(int j = 0; j < total_landmarks.size(); j++)
      {
        if(dist(p.x, p.y, double(total_landmarks[j].x_f), double(total_landmarks[j].y_f)) <= sensor_range)
        {
          available_landmarks_idx.insert(j);
        }
      }
      
      
      // find nearest neighbor
      double min_dist = DBL_MAX;
      int min_id = -1;
      for(const int& idx : available_landmarks_idx)
      {
        double distance = dist(map_obs.x, map_obs.y, double(total_landmarks[idx].x_f), double(total_landmarks[idx].y_f));
        if(distance < min_dist)
        {
          min_dist = distance;
          min_id = idx;
        }
        
      }
      map_obs.id = min_id;
           
      // Update weight if nearest neighbor is found; if not, do nothing
      if(map_obs.id == -1)
      {
        std::cout << "\n No neighbors have been found!";
      }
      else
      {
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double x_obs = map_obs.x;
        double y_obs = map_obs.y;
        double mu_x = double(total_landmarks[map_obs.id].x_f);
        double mu_y = double(total_landmarks[map_obs.id].y_f);
        // calculate normalization term
        double gauss_norm;
        gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
        

        // calculate exponent
        double exponent;
        exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                     + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

        // calculate weight using normalization terms and exponent
        double weight;
        weight = gauss_norm * exp(-exponent);
        p.weight *= weight;
        
          
      }
    }
    this->weights[i] = p.weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  discrete_distribution<int> distribution(this->weights.begin(), this->weights.end());
  
  vector<Particle> newParticles;
  vector<double> newWeights;
  
  for(int i = 0; i < num_particles; i++)
  {
    int rand_idx = distribution(gen);
    Particle newP = particles[rand_idx];
    newParticles.push_back(newP);
    newWeights.push_back(newP.weight);
  }
  this->particles = newParticles;
  this->weights = newWeights;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}