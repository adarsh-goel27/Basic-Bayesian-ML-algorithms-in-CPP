#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
using namespace std;

// Define a structure to hold a data point with features and a label.
struct DataPoint 
{
    vector<string> features;
    string label;
};

// Define the class for the Naive Bayes
class NaiveBayesClassifier 
{
private:
    // define map for storing the likelihood values for each feature value for different classes
    map<string, map<string, double>> likelihoods;
    // define map to store the class probabilities
    map<string, double> classProbabilities;
    // define set to store the unique classes
    set<string> uniqueClasses;
    // define total number of examples
    int examples;

public:
    void train( vector<DataPoint>& trainingData) 
    {
        // storing counts of each class and storing counts of each feature value for specific label
        for (auto dataPoint : trainingData) 
        {
            string label = dataPoint.label;
            uniqueClasses.insert(label);
            classProbabilities[label]++;

            for(int i = 0; i < dataPoint.features.size(); ++i )
                likelihoods[label][dataPoint.features[i] + "#" + to_string(i)]++;
        }

        // Calculate feature likelihoods
        for ( auto label : uniqueClasses) 
            for (auto& [featureValue, count] : likelihoods[label])
                count = count / classProbabilities[label];
               
        // Calculate class probabilities
        for (auto label : uniqueClasses) 
            classProbabilities[label] = classProbabilities[label] / trainingData.size();
        
        // caluclate total number of examples
        examples = trainingData.size();
        
                

    }

    vector<string> predict( DataPoint& dataPoint) 
    {
        double maxProbability = -1.0;
        string predictedClass;

        for (auto& label : uniqueClasses) 
        {
            double probability = classProbabilities[label];

            for(int i = 0; i < dataPoint.features.size(); ++i )
            {
                if (likelihoods[label].find(dataPoint.features[i] + "#" + to_string(i)) != likelihoods[label].end()) 
                    probability *= likelihoods[label][dataPoint.features[i] + "#" + to_string(i)];
                else 
                {
                    // Laplace smoothing for unseen features
                    probability *= (1.0 / examples) / (classProbabilities[label] + dataPoint.features.size() / examples);
                }
            }
            if (probability > maxProbability) 
            {
                maxProbability = probability;
                predictedClass = label;

            }
        }

        return vector<string> {predictedClass, to_string(maxProbability)};
    }
};

int main() {
    // Create a sample dataset for training
    vector<DataPoint> trainingData = 
    {
        {{"<=30", "high", "No", "fair"}, "No"},
        {{"<=30", "high", "No", "excellent"}, "No"},
        {{"31..40", "high", "No", "fair"}, "Yes"},
        {{">40", "medium", "No", "fair"}, "Yes"},
        {{">40", "low", "Yes", "fair"}, "Yes"},
        {{">40", "low", "Yes", "excellent"}, "No"},
        {{"31..40", "low", "Yes", "excellent"}, "Yes"},
        {{"<=30", "medium", "No", "fair"}, "No"},
        {{"<=30", "low", "Yes", "fair"}, "Yes"},
        {{">40", "medium", "Yes", "fair"}, "Yes"},
        {{"<=30", "medium", "Yes", "excellent"}, "Yes"},
        {{"31..40", "medium", "No", "excellent"}, "Yes"},
        {{"31..40", "high", "Yes", "fair"}, "Yes"},
        {{">40", "medium", "No", "excellent"}, "No"}
        
    };

    // defining Naive Bayes classifier 
    NaiveBayesClassifier classifier;
    classifier.train(trainingData);

    // Test the classifier with a new data point
    DataPoint testPoint = {{"<=30", "medium", "Yes", "fair"}, ""};
    vector <string> predictedClass = classifier.predict(testPoint);
    cout << "Predicted class: " << predictedClass[0]  << endl;
    cout << "Probability : " << predictedClass[1] << endl;

    return 0;
}