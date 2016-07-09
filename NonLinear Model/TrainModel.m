%% Train Model
% Use this script to train a linear or non-linear predictive model
%% Clear and Close Figures
clear ; close all; clc; more off

fprintf('Loading data ...\n');

%% Load Data


% Input your text file name in 'single quotes'. Each column should be
% complete (no NaN values) and headers/comments should be commented out
% with %. The parameters you want to fit to should be in the first columns
% and your output (y) values should be the final column

filename = 'testData.txt';

data = load(filename);
X = data(:, 1:end-1);
y = data(:, end);
m = length(y);

% Randomize order of training data and split into training, cross_validation, and test_error sets
randy = randperm(m);
X = X(randy,:);
y = y(randy,:);
train_rows = [1:round(0.7 * m)];
remaining = m - round(0.7 * m);
cv_rows = [round(0.7 * m)+1: round(0.7 * m) + round(remaining/2.0)];
test_rows = [round(0.7 * m) + round(remaining/2.0) + 1:m];
Xtest = X(test_rows,:);
Xcv = X(cv_rows,:);
X = X(train_rows,:);
ytest = y(test_rows,:);
ycv = y(cv_rows,:);
y = y(train_rows,:);
pause(0.5)
fprintf('Data loaded \n');

%% Train Multiple Models:
% This section will train multiple models on your data and select the best model
% based on error analysis

%% Map Features

% This parameter will map each each feature to each other feature to a
% polynomial degree of your choosing. For example, if 
% you choose deg = 2 and have 2 features to fit to this will return 
% [x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, x1^2*x2, x1^2*x2^2] If you have
% more features or higher degrees it will map each feature to each other in
% a similar manner. If this is too much, you can also decide not to map 
% features (deg = 0) or you can map whichever features you choose yourself
% and then input that for your data and choose deg = 0

% choose the different degrees, you want to try to fit by inputing them into 
% a horizontal vector;
deg = [1 2]; 

%% Alpha, Lambda, and Number of Iterations

% The alpha value chooses how 'far' the gradient descent 'jumps' each
% iteration. The smaller the more accurate, but the higher the
% number of necessary iterations and the longer the algorithm
% will take
alpha = .01;

% Number of iterations is how many times the gradient descent algorithm will run
% It needs to be high enough to converge towards 0, but not so high that this 
% script takes a million years to run. If your error ends up high, you can try 
% making this higher to see if it helps.
num_iters = 1000;


% The lambda value chooses how much 'penalty' you will place on the fitting 
% parameters. This can prevent over fitting (the model fits your data too well) 
% and fails to predict future inputs, but if you make it too high you 
% risk under fitting the data (the model doesn't fit well enough).

% choose the different lambda values you want to try to train on the data by 
% inputing them into a horizontal vector

lambda = [0 1 10];


number_of_models = length(deg) * length(lambda)
iteration = 1;
model_outcomes = cell(number_of_models, 6); % saves model parameters and errors {deg lambda mu sigma theta cv_error}

for degl = deg
  for lambdal = lambda
    fprintf('Traning Model %d out of %d: deg = %d, lambda = %.2f\n' , [iteration number_of_models degl lambdal])
    
    X_loop = mapFeatures(X,degl);
    Xcv_loop = mapFeatures(Xcv,degl);
    
    % Feature Normalize
    % This section 'normalizes' the features to make sure each has a mean of
    % around 0 and are of the same order of magnitude. This helps the algorithm
    % to run more efficiently.
    [X_loop mu sigma] = featureNormalize(X_loop);
    X_loop = [ones(size(X_loop,1),1) X_loop]; % add ones column for gradient descent
    
    % Gradient Descent
    theta = zeros(size(X_loop,2),1); % Initialize theta
    [theta, J_history] = gradientDescentMulti(X_loop, y, theta, alpha, num_iters, lambdal);
    
    % Plot J (cost) vs num_iter to make sure it's decreasing and reaching close
    % to 0:
    
    %figure; 
    %plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
    %xlabel('Number of iterations');
    %ylabel('Cost J');
    
    % Compute error on cross_validation set
    % Normalize your cross_validation set using the calculated mu and sigma
    Xcv_loop = [ones(size(Xcv_loop,1),1) Xcv_loop];
    for i = 1:size(Xcv_loop,1)
      for j = 2:size(Xcv_loop,2)
        Xcv_loop(i,j) = (Xcv_loop(i,j)-mu(j-1))/sigma(j-1);
      end
    end
    cv_error = computeCostMulti(Xcv_loop,ycv,theta,0); % Compute error on cross_validation set
    
    % Save variables to model_outcomes cell
    model_outcomes{iteration, 1} = degl;
    model_outcomes{iteration, 2} = lambdal;
    model_outcomes{iteration, 3} = mu;
    model_outcomes{iteration, 4} = sigma;
    model_outcomes{iteration, 5} = theta;
    model_outcomes{iteration, 6} = cv_error;
    iteration = iteration + 1;
  end
end

% Select the model with the lowest error
[value row] = min([model_outcomes{:,6}]');

% Note for programmer: {deg lambda mu sigma theta cv_error}
deg = model_outcomes{row,1};
save deg.mat deg
lambda = model_outcomes{row,2};
mu = model_outcomes{row,3};
save mu.mat mu
sigma = model_outcomes{row,4};
save sigma.mat sigma
trainedTheta = model_outcomes{row,5};
save trainedTheta.mat trainedTheta
cv_error = model_outcomes{row,6}; 

Xtest = [ones(size(Xtest,1),1) mapFeatures(Xtest,deg)]; %Map Xtest and add column of ones

% Normalize your Xtest set using the calculated mu and sigma
for i = 1:size(Xtest,1)
  for j = 2:size(Xtest,2)
    Xtest(i,j) = (Xtest(i,j)-mu(j-1))/sigma(j-1);
  end
end
test_error = computeCostMulti(Xtest,ytest,trainedTheta,0); %compute error

fprintf('The program has selected the optimal model with parameters\ndeg = %d\nlambda = %.3f\nand an error of %.3f\n\nIf this model seems acceptable, you can now use it to predict new outcomes using predict(X),\nwhere X is a row vector or matrix of row vectors with containing the paramaters who outcome you wish to predict.\nBe sure that the parameter(s) in your input vector are in the same order as the vectors your model was trained on.', [deg lambda test_error])

