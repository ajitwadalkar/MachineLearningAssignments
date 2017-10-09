%%  Linear Regression

% 
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     
%


%% Initialization
clear ; close all; clc


%% =================== Part 1: Symbolic variable method ===================

clear all;

data = load('HM1data1.txt');

% The data set is 2-dimensional:

% The 1st column, X, is the independent variable,
% and it refers to the population zise in 10,000s

% The second column, y, is the dependent variable, and it y refers to the
% profit in $10,000s

X = data(:, 1); % 1st column of data
y = data(:, 2); % 2nd column of data

% m: the size, i.e., the number of training samples
m = length(X(:,1)); % the length of the first column of X

%% Start regression here
% extend the data set by the bias column: 
%       Each row receives a 1 in front

eX=[ones(m,1) X];

% size of a data instance in the extended data set: 
%       length of the first row of eX, which is 1 + length of 1st row of X

nPlusOne = length(eX(1,:));  % we really do not need this explicitely...

% Now set up the weight vector w:

syms w0 w1;

% Put them in a column vector
w=[w0; w1];

% inspect sizes of eX(1,:) and w
size(eX(1,:));
size(w);

% Define now the linear hypothesis
for l=1:m, hX(l) = eX(l,:)*w; end
    
% Define now the cost function / error function

Jw = sum((hX - y').^2); % dot notation means elementwise operation for vectors

% use matlab function 'gradient' 
grad=gradient(Jw);

%Inspect S
size(grad);

% Use Matlab function 'solve' to solve the equationa obtained by setting 
% gradient = 0 
[w0, w1]=solve(grad(1), grad(2));

% print W to screen
fprintf('W found by symbolic variable method: ');
fprintf('%f %f \n', double(w0), double(w1));

% Now plug the solutions into hX: use Matlab function 'eval'
eval(hX);

%% PLOT RESULTS

plotData(X,y)

hold on; % holds the figure in order to be able to plot on it successively

plot(X', eval(hX), '-');

%%  NOW THE MATRIX BASED SOLUTION

% Set up quantities

jTemp = eX*w - y;

J2 = 1/2*jTemp'*jTemp;
grad2 = gradient(J2);

% check:
grad22=eX'*eX*w - eX'*y;

[w0, w1]=solve(grad22);

w=inv(eX'*eX)*eX'*y;

clear all;

%% =================== Part 2: Gradient descent method ===================
fprintf('Plotting Data ...\n')

% read in the data set
data = load('HM1data1.txt');

X = data(:, 1); % 1st column of data
y = data(:, 2); % 2nd column of data

m = length(y); % number of training examples

% Write the fhe function Plot Data, which  plots y against X

plotData(X, y);

% %% Dataset 1
% % X is the data set : 3 x 1 column vector
% % each row in the data set is a traning instance
% X=[1; 2; 3];
% 
% % y is the output variable: each row is the output for the corresponding
% % training instance;
% % y is a 3 x 1 vector
% y=[2;1;3];

fprintf('Running Gradient Descent ...\n')

% Extend X by adding a column of ones to the data matrix
X = [ones(m, 1), data(:,1)];

% initialize the weight vector W 
W = zeros(2, 1); 

% Set gradient descent settings: Note that in class, I actually used Matlab
% symolic variables to calculate directly the solution of the optimization
% problem.

% Here we use a NUMERICAL method by recomputing the gradient 
% We set some quantities necessary to do this

iterations = 1500;
alpha = 0.01;

% Compute and display initial cost
computeCost(X, y, W);

% Next we need to run the function gradientDescent.
% It returns the solution for W
W = gradientDescent(X, y, W, alpha, iterations);

% print W to screen
fprintf('W found by gradient descent method: ');
fprintf('%f %f \n', W(1), W(2));

% Plot the linear fit
hold on; % keep previous plot visible

% Recall that the 1st column of X is made up of 1's.  So we need to plot
% against the 2nd column.  Also, the hypothesis is X*W, where * is matrix
% multiplication

plot(X(:,2), X*W, '-'); 
legend('Training data', 'Linear regression','Descent Data', 'Gradient Descend')
hold off   % I this moment, release the figure, so no other plots are made on it

%% Now we use the model to make predictions

% 1. Predict values for population sizes of 35,000 

predict1 = [1, 3.5]*W;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);

% 2. Predict values for population of size 70,000
predict2 = [1, 7]*W;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);


%% ============= Part 3: Visualizing the cost function J(w_0, w_1) =============

fprintf('Visualizing J(w_0, w_1) ...\n')

% Grid over which we will calculate J
W0_vals = linspace(-10, 10, 100);
W1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(W0_vals), length(W1_vals));

% Fill out J_vals
for i = 1:length(W0_vals)
    for j = 1:length(W1_vals)
	  t = [W0_vals(i); W1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
subplot(1,2,1)
surf(W0_vals, W1_vals, J_vals)
xlabel('w_0'); ylabel('w_1');
title('Surface')

% Contour plot
subplot(1,2,2)

% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(W0_vals, W1_vals, J_vals, logspace(-2, 3, 20))
xlabel('W_0'); ylabel('W_1');
hold on;
plot(W(1), W(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title('Contour')