#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#define M_PI 3.14159265358979323846
using namespace std;

// defining the datapoint struct
struct DataPoint 
{
    vector<double> values;
};

// function to display the matrix
void displayMatrix( vector<vector<double>>& matrix) 
{
    for (auto row : matrix) 
    {
        for (double cell : row) 
            cout << cell << " ";
        cout << endl;
    }
}

// Function to parse a CSV file and store the data in a matrix
vector<vector<double>> parseCSVtoMatrix(string filename) 
{
    ifstream file(filename);
    vector<vector<double>> matrix;

    string line;
    while (getline(file, line)) 
    {
        vector<double> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) 
            row.push_back(stod(cell));

        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

// defining the EM clustering class
class EMClustering 
{
private:
    int numClusters;
    int numFeatures;
    int numIterations;
    vector<DataPoint> dataPoints;
    vector<vector<double>> means;
    vector<vector<double>> covariances;
    vector<double> weights;

public:
    EMClustering(int k, int features, int iterations): numClusters(k), numFeatures(features), numIterations(iterations) 
    {
       string filenameX = "iris.csv"; // Change this to your CSV file
       vector<vector<double>> dataX = parseCSVtoMatrix(filenameX);
       int examples = dataX.size();
       for(int i = 0; i < examples; ++i)
       {
            DataPoint data;
            data.values = dataX[i];
            dataPoints.push_back(data);
       }
    }

    

    void initialize() 
    {
        // Initialize means and covariances randomly
        mt19937 gen(-1851021163);
        uniform_real_distribution<double> distribution(0.0, 10.0); // Assuming data in range [0, 10]

        for (int i = 0; i < numClusters; ++i) 
        {
            vector<double> clusterMean;
            vector<double> clusterVariance;

            for (int j = 0; j < numFeatures; ++j) 
            {
                clusterMean.push_back(distribution(gen));
                clusterVariance.push_back(1.0); // Assume spherical covariance initially
            }
            means.push_back(clusterMean);
            covariances.push_back(clusterVariance);
            weights.push_back(1.0 / numClusters); // Equal weights for each cluster initially
        }
    }

    double gaussianPDF(vector<double>& x, vector<double>& mean, vector<double>& variance) 
    {
        double pdf = 1.0;
        for (int i = 0; i < numFeatures; ++i) 
            pdf *= exp(-0.5 * pow((x[i] - mean[i]), 2) / variance[i]) / sqrt(2 * M_PI * variance[i]);
        return pdf;
    }

    void run() 
    {
        for (int iter = 0; iter < numIterations; ++iter) 
        {
            // E-step - Expectation step
            vector<vector<double>> responsibilities(dataPoints.size(), std::vector<double>(numClusters));
            for (int i = 0; i < dataPoints.size(); ++i) 
            {
                double totalProb = 0.0;
                for (int k = 0; k < numClusters; ++k) 
                {
                    responsibilities[i][k] = weights[k] * gaussianPDF(dataPoints[i].values, means[k], covariances[k]);
                    totalProb = totalProb + responsibilities[i][k];
                }
                for (int k = 0; k < numClusters; ++k) 
                    responsibilities[i][k] = responsibilities[i][k] / totalProb; // Normalize responsibilities
                
            }

            // M-step - Maximization step
            for (int k = 0; k < numClusters; ++k) 
            {
                vector<double> weightSum(numFeatures, 0.0);
                vector<double> mean(numFeatures, 0.0);
                vector<double> variance(numFeatures, 0.0);

                for (int  i = 0; i < dataPoints.size(); ++i) 
                {
                    for (int j = 0; j < numFeatures; ++j) 
                    {
                        weightSum[j] =weightSum[j] + responsibilities[i][k];
                        mean[j] = mean[j] + responsibilities[i][k] * dataPoints[i].values[j];
                    }
                }

                for (int j = 0; j < numFeatures; ++j) 
                {
                    weights[k] = weightSum[j] / dataPoints.size();
                    means[k][j] = mean[j] / weightSum[j];
                }

                for (int i = 0; i < dataPoints.size(); ++i) 
                {
                    for (int j = 0; j < numFeatures; ++j) 
                        variance[j] = variance[j] + responsibilities[i][k] * pow(dataPoints[i].values[j] - means[k][j], 2);
                    
                }

                for (int j = 0; j < numFeatures; ++j) 
                    covariances[k][j] = variance[j] / weightSum[j];

            }
        }
    }

    void displayResults() 
    {
        cout << "Cluster means and covariances:" << endl;;
        for (int k = 0; k < numClusters; ++k) 
        {
            cout << "Cluster " << k << " - Mean: (";
            for (int i = 0; i < numFeatures; ++i) 
            {
                cout << means[k][i];
                if (i != numFeatures - 1) 
                    cout << ", ";
                
            }
            cout << ") - Covariance: (";
            for (int i = 0; i < numFeatures; ++i) 
            {
                cout << covariances[k][i];
                if (i != numFeatures - 1) 
                    cout << ", ";
                
            }
            cout << ")" << endl;
        }
    }
};

int main() 
{
    EMClustering em(3, 4, 10000); //  clusters,  features, and  iterations
    em.initialize();
    em.run();
    em.displayResults();

    return 0;
}
