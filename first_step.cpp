
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>


// this will calculate the euclidean distance
double euclidean(const std::vector<double>& data_1, const std::vector<double>& data_2) {
    double sum_ = 0;
    for (size_t i = 0; i < data_1.size() - 1; i++) {
        double dis = std::pow(data_1[i] - data_2[i], 2);
        sum_ += dis;
    }
    return std::sqrt(sum_);
}

//calculates the kNN for a certain data point with all other points in the dataset
std::vector<int> K_nn(const std::vector<double>& data_point, const std::vector<std::vector<double>>& original_dataset, int k) {
    std::vector<std::pair<int, double>> distance;
    for (size_t idx = 0; idx < original_dataset.size(); idx++) {
        double dis = euclidean(data_point, original_dataset[idx]);
        distance.push_back(std::make_pair(idx, dis));
    }
    std::sort(distance.begin(), distance.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    std::vector<int> ans;
    for (int i = 0; i < k; i++) {
        int pos = distance[i].first;
        ans.push_back(pos);
    }
    return ans;
}


//created a vectors with index as its index in the dataset and the index of its reverse nearest neighbors
std::vector<std::vector<int>> reverse_NN(const std::vector<std::vector<double>>& X) {
    std::vector<std::vector<int>> indices;
    for (size_t idx = 0; idx < X.size(); idx++) {
        std::vector<int> temp = K_nn(X[idx], X, 3);
        indices.push_back(temp);
    }
    std::vector<std::vector<int>> rnn(X.size(), std::vector<int>());
    for (size_t i = 0; i < indices.size(); i++) {
        rnn[indices[i][1]].push_back(i);
    }
    return rnn;
}

std::vector<std::vector<double>> generate_synthetic_points(const std::vector<std::vector<double>>& dataset, const std::vector<std::vector<int>>& rnn, int k, int num_synthetic_points, double alpha) {
    std::vector<std::vector<double>> synthetic_points;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t p = 0; p < dataset.size(); p++) {
        if (rnn[p].empty()) {
            continue;
        }
        for (int _ = 0; _ < num_synthetic_points; _++) {
            int neighbor = rnn[p][std::uniform_int_distribution<int>(0, rnn[p].size() - 1)(gen)];
            std::vector<double> synthetic_point;
            for (size_t i = 0; i < dataset[p].size() - 1; i++) {
                double synthetic_value = dataset[p][i] + alpha * (dataset[neighbor][i] - dataset[p][i]);
                synthetic_point.push_back(synthetic_value);
            }
            double label = dataset[p][4];
            synthetic_point.push_back(label);
            synthetic_points.push_back(synthetic_point);
        }
    }
    return synthetic_points;
}


struct DataPoint {
    double x;
    double y;
    double label;
};

std::vector<DataPoint> generateDataset(int numEntries) {
    std::vector<DataPoint> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int i = 0; i < numEntries; i++) {
        double randomX = dist(gen);
        double randomY = dist(gen);
        double label = (randomX >= randomY) ? 1.0 : 0.0;
        dataset.push_back({ randomX, randomY, label });
    }

    return dataset;
}


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 100.0);
    std::vector<std::vector<double>> groundTruth;
    int numEntries = 30;
    //srand(time(NULL));
    for (int i = 0; i < numEntries; i++) {
        double randomX = dist(gen);
        double randomY = dist(gen);
        double label = (randomX >= randomY) ? 1.0 : 0.0;
        groundTruth.push_back({ randomX, randomY, label});
    }
    for (auto point : groundTruth) {
        for (auto value : point) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    //for (const auto& dataPoint : groundTruth) {
      //  std::cout << "X: " << dataPoint.x << ", Y: " << dataPoint.y << ", Label: " << dataPoint.label << std::endl;
    //}

    int k = 4;
    int num_synthetic_points = 2;
    double alpha = 0.5;

    //std::vector<std::vector<double>> initial_dataset = { /* your initial dataset here */ };

    std::vector<std::vector<int>> rnn = reverse_NN(groundTruth);

    std::vector<std::vector<double>> synthetic_points_pca = generate_synthetic_points(groundTruth, rnn, k, num_synthetic_points, alpha);

    for (const auto& point : synthetic_points_pca) {
        for (const auto& value : point) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << synthetic_points_pca.size() << std::endl;

    return 0;
}
