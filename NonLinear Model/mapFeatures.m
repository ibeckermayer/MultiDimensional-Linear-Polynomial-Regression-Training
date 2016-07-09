function out = mapFeatures(X, deg)

% Maps each feature X_i to each other feature X_i+1, X_i+2, ... X_n
% to a dim degree polynomial (and adds a column of ones)

out = [];

for i = 1:deg
    out = [out X.^i];
end

n = size(X,2);
for i = 1:n
    for j = (i+1):n
        out = [out mapFeature(X(:,i),X(:,j),deg)];
    end
end

if deg == 0
    out = X;
end

end

