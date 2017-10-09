function J = computeCost(X, y, W)
m  = length(y); 
hX = zeros(m,1);
J  = 0;
h  = X * W;
hX = (h - y).^2;            
J  = 1/(2*m) * sum(hX);     

end
