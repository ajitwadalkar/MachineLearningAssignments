function [W, J_history] = gradientDescent(X, y, W, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
WLen = length(W);
tempVal = W;
for iter=1:num_iters
    temp = (X*W - y);
    for i=1:WLen
        tempVal(i,1) = sum(temp.*X(:,i));
    end
    W = W - (alpha/m)*tempVal;    
    J_history(iter) = computeCost(X,y,W);
end

end
