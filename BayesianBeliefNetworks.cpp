#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <math.h>

using namespace std;

// Node represents a variable in the Bayesian Network
struct Node 
{
    string name;
    vector<Node*> parents;
    map<string, double> probabilities;
};

double variableElimination(map<string, string>& evidence, map<string, Node*>& network) 
{
    double probability = 1.0;
    for(auto entry : network)
    {
        Node* currentNode = entry.second;
        if((currentNode -> parents).size() == 0)
        {
            string event = evidence[currentNode -> name];
            probability *= currentNode -> probabilities[event];
        }
        else
        {
            string event = "";
            for(auto parent : currentNode -> parents)
                event += evidence[parent -> name] + ",";
            event = event.substr(0, event.size() - 1);
            if(evidence[currentNode -> name] == "true")
                probability *=  currentNode -> probabilities[event];
            else
                 probability *=  1 - ( currentNode -> probabilities[event]);
        }
   }
   return probability;

}
double generalQuery(map<string, string>& query, map<string, Node*>& network)
{
    double probability = 0.0;
    vector<string> unknowns;
    for(auto entry : query)
    {
        if(entry.second == "-")
            unknowns.push_back(entry.first);
        
    }
    if(unknowns.size() == 0)
        return variableElimination(query,network);

    for(int i = 0; i < pow(2,unknowns.size()); ++i)
    {
        for(int j = 0; j < unknowns.size();++j)
        {
            if((i >> j) % 2  == 0)
                query[unknowns[j]] = "false";
            else 
                query[unknowns[j]] = "true";
        }
        
        probability += variableElimination(query,network);
    }
    return probability;
}


int main() 
{
    // Creating nodes for the Bayesian Network
    Node A = {"Buglary"};
    Node B = {"Earthquake"};
    Node C = {"Alarm"};
    Node D = {"David Calls"};
    Node E = {"Sophia Calls"};

    // Define relationships between nodes
    C.parents.push_back(&A);
    C.parents.push_back(&B);
    D.parents.push_back(&C);
    E.parents.push_back(&C);

    // Define probabilities (conditional probability distributions) for each node
    A.probabilities["false"] = 0.001;
    A.probabilities["false"] = 0.999;

    B.probabilities["true"]= 0.002;
    B.probabilities["false"] = 0.998;

    // Conditional Probability Distribution for C given A and B
    C.probabilities["true,true"] = 0.95;
    C.probabilities["true,false"] = 0.94;
    C.probabilities["false,true"] = 0.29;
    C.probabilities["false,false"] = 0.001;

    // Conditional Probability Distribution for D given C
    D.probabilities["true"] = 0.95;
    D.probabilities["false"] = 0.05;

    // Conditional Probability Distribution for E given C
    E.probabilities["true"] = 0.8;
    E.probabilities["false"] = 0.01;

    // Create a network by mapping node names to their respective nodes
    map<string, Node*> network;
    network["A"] = &A;
    network["B"] = &B;
    network["C"] = &C;
    network["D"] = &D;
    network["E"] = &E;

    // Perform inference - Querying variable D given evidence A=false, B=true
    map<string, string> query = {{"Buglary", "false"}, {"Earthquake", "false"}, {"Alarm", "true"}, {"David Calls", "true"}, {"Sophia Calls", "-"}};
    double ans = generalQuery( query, network);
    cout << "Probability of the query: " << ans <<  endl;
    return 0;
}
