function prediction = predict(X)
% Predict new outcomes using predict(X), where X is a row vector or matrix of 
% row vectors with containing the paramaters who outcome you wish to predict.
% Be sure that the parameter(s) in your input vector are in the same order as 
% the vectors your model was trained on.

load deg.mat; % load deg in order to map input vector properly to your model
load trainedTheta.mat % load your trained theta vector
load mu.mat % load mu
load sigma.mat % load sigma

% Should be a horizontal vector

X = [ones(size(X,1),1) mapFeatures(X,deg)]; %Map X and add column of ones
size(X)
size(trainedTheta)
%Normalize Vector

for i = 1:size(X,1)
  for j = 2:size(X,2)
    X(i,j) = (X(i,j)-mu(j-1))/sigma(j-1);
  end
end

price = X * trainedTheta;

fprintf(['Predicted Outcome:\n %f\n'], price);

