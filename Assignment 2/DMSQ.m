function [X] = DMSQ(X)
 x = X(:,2:end);
 x= x.^2;
X = [X,x];
end