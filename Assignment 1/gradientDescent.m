function [W, J_history] = gradientDescent(X, y, W, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn W
%   W = GRADIENTDESCENT(X, y, W, alpha, num_iters) updates by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % declare J_history and initialize to 0

for iter = 1:num_iters

	% ====================== YOUR CODE HERE ======================
	% Perform a single gradient step on the parameter vector W

	x = X(:,2);
	h = W(1) + (W(2) * x);
	error = h - y; % m x 1 vector: unsquared difference hypothesis - y

	% X is m x n matrix, so to multiply by errors we need to transpose it
	% that is, X'*error
	% then scale / multiply by alpha and (1/m)
	% the sum from the formula for updating W 
	% is autmatically taken care by the matrix multplication X'*error 
	 
	change_W1 = sum(error) * (alpha/m);
	change_W2 = sum(x.*error) * (alpha/m);

	% update W
	W(1) = W(1) - change_W1;
	W(2) = W(2) - change_W2;

	% compute cost for the new W
	J = computeCost(X, y, W);

	% Save the cost J in every iteration    
	J_history(iter) = J; %save current iteration cost

end %iter

end % function
