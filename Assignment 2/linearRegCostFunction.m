function [J, grad] = linearRegCostFunction(X, y, W, lambda)
m = length(y); 
J = 0;
grad = zeros(size(W));
h = X*W;
W_reg = [0;W(2:end, :);];
J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*W_reg'*W_reg;
grad = (1/m)*(X'*(h-y)+lambda*W_reg);
grad = grad(:);

end