function [W] = L2Reg(X, y, W, alpha, num_iters, Lambda)
m = length(y); 
for iter = 1:num_iters
W_temp=W*Lambda;
W_temp(1)=0;
W = W - (alpha/m)*((X')*(X*W - y)+W_temp);
end 
end 
