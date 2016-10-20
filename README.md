# MultiDimensional-Linear-Polynomial-Regression-Training
N-Dimensional training and prediction using gradient descent in Matlab

Most of the code is original, however I borrowed the templates from Andrew Ng's Coursera based Machine Learning course for computeCostMulti and gradientDescentMulti.

The classifier can be run using TrainModel.m. Before running, make sure you have your training data in a text file, the first column's being your input parameters and the last column being your output.

TrainModel takes your input data, shuffles it and creates a training set, test set, and cross validation set. Be sure to read through the TrainModel code and set the parameters you want to try on the model, including the degree polynomial you want to try to fit with.

During training, the code outputs every thousand iterations so you can see how fast it's running. Training can sometimes be computationally intensive depending on how many iterations you are doing and to what degree polynomial you are fitting with. This effect is amplified since all the code is 'home-made' and the algorithms are not optimized or GPU boosted.

TrainModel will train multiple models on your data, and output the model with the lowest error. The parameters for this model will be saved in trainedTheta.mat (along with mu and sigma which are necessary for normalization when making predictions).

Once the model has been trained, you can use predict.m to predict a new output given one or more input vectors.
