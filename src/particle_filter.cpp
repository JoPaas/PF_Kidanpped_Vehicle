/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// random engine for particle state noise
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	// particle state noise
	normal_distribution<double> x_noise(0, std[0]);
	normal_distribution<double> y_noise(0, std[1]);
	normal_distribution<double> theta_noise(0, std[2]);

	for (auto i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		// add noise to state
		p.x = x + x_noise(gen);
		p.y = y + y_noise(gen);
		p.theta = theta + theta_noise(gen);
		// init weight
		p.weight = 1.0;
		// add particle
		particles.push_back(p);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// measurement noise
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);
	// yaw_rate threshold
	double eps = 0.0001;

	for (auto i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < eps) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add process noise
		particles[i].x += x_noise(gen);
		particles[i].y += y_noise(gen);
		particles[i].theta += theta_noise(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double default_dist = numeric_limits<double>::max();

	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs o = observations[i];
		// init landmark distance and mapping id
		double min_dist = default_dist;
		int map_id = -1;
		// loop over predictions
		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs p = predicted[j];
			// calclate distance o p
			double curr_dist = dist(o.x, o.y, p.x, p.y);
			//if smaller than min_dist, update mapping
			if (curr_dist < min_dist) {
				map_id = p.id;
				min_dist = curr_dist;
			}
		}
		// associate prediction to observation
		if (map_id == -1) cout << "we have a problem" << endl;
		//cout << "map_id: " << map_id << endl;
		observations[i].id = map_id; // not o.id because thats a member variable
		// what if multiple observations are associated to the same observation?
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (auto i = 0; i < num_particles; i++) {
		// current surrounding of vehicle (in sensor FOV)
		vector<LandmarkObs> surrounding;
		// state of particle
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;
		//cout << "particle_pos: [" << x_p << ", " << y_p << "]" << endl;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			int id_lm = map_landmarks.landmark_list[j].id_i;
			float x_lm_f = map_landmarks.landmark_list[j].x_f;
			float y_lm_f = map_landmarks.landmark_list[j].y_f;
			// cast to double
			double x_lm = (double) x_lm_f;
			double y_lm = (double) y_lm_f;
			//cout << "comp: " << x_lm_f << " " << x_lm << endl;
			// form surrounding from landmarks in range
			if (dist(x_p, y_p, x_lm, y_lm) <= sensor_range) {
				//cout << "surr_point: [" << x_lm << ", " << y_lm << "]" << endl;
				surrounding.push_back(LandmarkObs{id_lm, x_lm, y_lm});
			}
		}

		// transform current observations in map frame
		vector<LandmarkObs> obs_map;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double x_veh = observations[j].x;
			double y_veh = observations[j].y;
			// transform
			double x_map = x_veh * cos(theta_p) - y_veh * sin(theta_p) + x_p;
			double y_map = x_veh * sin(theta_p) + y_veh * cos(theta_p) + y_p;
			obs_map.push_back(LandmarkObs{observations[j].id, x_map, y_map});
		}

		// associate
		dataAssociation(surrounding, obs_map);
		// reset particle weight to 1.0 for weight update
		particles[i].weight = 1.0;
		for (unsigned int j = 0; j < obs_map.size(); j++) {
			double x_om = obs_map[j].x;
			double y_om = obs_map[j].y;
			int match = obs_map[j].id;
			// placeholders for matched predictions coordinates
			double x_mp, y_mp;
			for (unsigned int k = 0; k < surrounding.size(); k++) {
				if (surrounding[k].id == match) {
					x_mp = surrounding[k].x;
					y_mp = surrounding[k].y;
				}
			}
			//cout << "observation: [" << x_om << ", " << y_om << "] assoc_landmark: [" << x_mp << ", " << y_mp << "]" << endl;
			// calculate weight of observation
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			// split up calculation for better readability
			double inner_1 = pow(x_mp - x_om, 2) / (2.0 * pow(sig_x, 2));
			double inner_2 = pow(y_mp - y_om, 2) / (2.0 * pow(sig_y, 2));
			double exponent = exp(-(inner_1 + inner_2));
			//cout << "exponent: " << exponent << endl;
			double weight_obs = (1 / (2 * M_PI * sig_x * sig_y)) * exponent;
			//cout << "observation weight: " << weight_obs << endl;
			// update particle weight starting at 1.0
			particles[i].weight *= weight_obs;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled;
	vector<double> weights;

	for (auto i = 0; i < num_particles; i++) weights.push_back(particles[i].weight);
	/*cout << "RESAMPLE" << endl;
	cout << "weights size: " << weights.size() << endl;
	cout << "weights: ";
	for (auto weight : weights) cout << weight << ", ";
	cout << endl;*/

	// resampling wheel code
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto ind = uniintdist(gen);

	//maximum weight
	double max_weight = *max_element (weights.begin(), weights.end());
	//cout << "max weight: " << max_weight << endl;
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	// step beta
	double beta = 0.0;
	for (auto i = 0; i < num_particles; i++) {
		// move by 2.0 times random weight
		beta += 2.0 * unirealdist(gen);
		// increase index until beta is met
		while (beta > weights[ind]) {
			beta -= weights[ind];
			ind = (ind + 1) % num_particles;
		}
		// add sampled particle
		//cout << "resampling particle: " << ind << endl;
		resampled.push_back(particles[ind]);
	}
	// update resampled particles
	particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
