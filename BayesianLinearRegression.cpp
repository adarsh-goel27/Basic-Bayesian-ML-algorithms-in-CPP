#include <iostream>
#include "eigen\Eigen\Eigen"
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace Eigen;
using namespace std;

// Define the Bayesian Linear Regression class
class BayesianLinearRegression 
{
private:
    MatrixXd X;   // Input  matrix consisting of feature value
    VectorXd y;   // Target values
    double alpha; // Precision parameter for the prior
    VectorXd w_mean; // Mean of the weight distribution
    MatrixXd w_cov;  // Covariance of the weight distribution

public:
    // Constructor
    BayesianLinearRegression(MatrixXd& X, VectorXd& y, double alpha) : X(X), y(y), alpha(alpha) 
    {
        int num_samples = X.rows();
        int num_features = X.cols();
        w_mean = VectorXd::Zero(num_features);
        w_cov = alpha * MatrixXd::Identity(num_features, num_features);
    }

    void fit() 
    {
        // Bayesian update of the weight distribution
        MatrixXd A =  X.transpose() * X + w_cov.inverse();
        VectorXd b = X.transpose() * y;
        w_mean = A.ldlt().solve(b);
        w_cov = A.inverse();
    }

    vector<double> predict(VectorXd& x) 
    {
        // Predict the mean and variance of the target variable
        double mean = x.dot(w_mean);
        double variance = 1.0 / alpha + x.dot(w_cov * x);
        return vector<double>({mean, variance});
    }
};
// Function to display the matrix
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
vector<vector<double>> parseCSVtoMatrix( string& filename) 
{
    ifstream file(filename);
    vector<vector<double>> matrix;

    if (file.is_open()) 
    {
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
    } 
    else 
        cerr << "Unable to open the file." << endl;
    

    return matrix;
}
MatrixXd convertToMatrixXd(vector<vector<double>>& data)
{
    // Convert the vector of vectors to an Eigen MatrixXd
    MatrixXd X(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        X.row(i) = Map<VectorXd>(data[i].data(), data[i].size());
    return X;
}
VectorXd convertToVectorXd(vector<vector<double>>& data)
{
    VectorXd y(data.size());
    for (int i = 0; i < data.size(); ++i) 
        y.coeffRef(i) = data[i][0]; 
    
    return y;
}


int main() 
{
    string filenameX = "dataX.csv"; // Change this to your CSV file
    string filenameY = "dataY.csv"; // Change this to your CSV file
    vector<vector<double>> dataX = parseCSVtoMatrix(filenameX);
    vector<vector<double>> dataY = parseCSVtoMatrix(filenameY);
    MatrixXd X = convertToMatrixXd(dataX);
    VectorXd y = convertToVectorXd(dataY);
    // Create a Bayesian Linear Regression model
    double alpha = 9.0; // Precision parameter for the prior
    BayesianLinearRegression regression(X, y, alpha);

    // Fit the model
    regression.fit();

    // Make predictions
    VectorXd new_data_point(8);
    new_data_point << -122.23,37.88,41,880,129,322,126,8.3252; // Replace with your new data point
    vector<double> prediction = regression.predict(new_data_point);

    cout << "Predicted mean: " << prediction[0] << endl;
    cout << "Predicted variance: " << prediction[1] << endl;
    return 0;
}
