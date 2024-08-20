
#ifndef KNN_H
#define KNN_H

#include <vector>

std::vector<int> K_nn(const std::vector<double>& point, const std::vector<std::vector<double>>& dataset, int k);
std::vector<std::vector<double>> generate_synthetic_points(const std::vector<std::vector<double>>& dataset, const std::vector<std::vector<int>>& rnn, int k, int num_synthetic_points, double alpha);
double euclidean(const std::vector<double>& data_1, const std::vector<double>& data_2);
std::vector<std::vector<int>> reverse_NN(const std::vector<std::vector<double>>& X);


#endif
