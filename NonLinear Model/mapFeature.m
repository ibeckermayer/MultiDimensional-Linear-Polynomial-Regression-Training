function out = mapFeature(X1, X2, deg)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2, deg) maps the two input features
%   to quadratic features of the deg-th degree used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1*X2, X1*X2.^2, ... ,X1*X2.^deg 
%
%   Inputs X1, X2 must be the same size
%

degree = deg;
out = [];
for i = 1:degree
    for j = 1:degree
        out(:, end+1) = (X1.^i).*(X2.^j);
    end
end

end