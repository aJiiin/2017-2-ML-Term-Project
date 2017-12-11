Predicting Energy Efficiency
========================
About
-------
This model is a logistic regression model using softmax function that classify each heating load and cooling load's category

Requirement
--------------
* tensorflow 1.2
* pythone 3.5.x (I didnt't use anaconda)
* numpy
* matplotlib

Data Set
----------------------
Energy Efficiency Data Set from UCI Machine Learning Repository
<http://archive.ics.uci.edu/ml/datasets/Energy+efficiency>

Architecture
-------------
The neural network, which has 8 input parameters and 1 output value, consisits of five layers.
For weight vaule in each layer, I used xavier initializer and the number of hidden nodes is 50, 20, 30, 70 for each hidden layer.
Also, except output layer, each layer uses ReLU function instead of sigmoid function for better performance.
Output layer uses softmax function and one-hot to classify one category which each y1 and y2 belongs to.
Optimizer that I used for this model is an AdamOptimizer and learning rate is 0.001 for heating load, 0.0001 for cooling load. 
For y1 value, which means heating load, there are four categories and for y2 value, which means cooling load, there are five categories.