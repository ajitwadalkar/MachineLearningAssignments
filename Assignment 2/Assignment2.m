clear ;
close all;
clc;

%Data Import
DataMat = importdata('Data.mat');

%DataMat dimentions
[DMSize,DMAttrib]=size(DataMat);
DataMatX = DataMat(:,1:(DMAttrib-1));
DataMatY = DataMat(:,DMAttrib:DMAttrib);

%DataMat Normalization
DMMean = mean(DataMatX);
DMStd = std(DataMatX);
DMRepStd = repmat(DMStd,DMSize,1);
DMRepMean = repmat(DMMean,DMSize,1);
NormMat = (DataMatX-DMRepMean)./DMRepStd;
NormMat=[NormMat,DataMatY];

%Adding Ones
NormMat=[ones(DMSize,1) NormMat];
[DMSize,DMAttrib]=size(NormMat);

%Dividing DataMat into Training and Testing Datasets
TrDSize = fix(DMSize/2);
TstDSize = DMSize - TrDSize;
RandIndex = randperm(DMSize);
TrDMat = NormMat(RandIndex(1:TrDSize),:);
TstDMat = NormMat((TrDSize+1):DMSize,:);

%Dividing Training Data into X and Y
TrDMatX = TrDMat(:,1:(DMAttrib-1));
TrDMatY = TrDMat(:,DMAttrib:DMAttrib);

%Dividing Tst Data into X and Y
TstDMatX = TstDMat(:,1:(DMAttrib-1));
TstDMatY = TstDMat(:,DMAttrib:DMAttrib);

%Calculation of Weighted vectors for certain lambda values
iterations = 1500;
alpha =0.01;
W = zeros(4,1);
W_lambda = [];
Lambdacost = [];

Lambda = logspace(-3,-0.5,15);
for i=1:length(Lambda)
    W_lambda = [W_lambda,L2Reg(TrDMatX, TrDMatY, W, alpha, iterations,Lambda(i))];
    Lambdacost = [Lambdacost,linearRegCostFunction(TrDMatX, TrDMatY, W_lambda(:,i),Lambda(i))];
end
[LM,LI] = min(Lambdacost);
Min_Lambda = Lambda(LI)
W_lambda = W_lambda';
SumofLambda = sum(W_lambda);
[MinVal,MinIndex] = min(SumofLambda);

%Regularized Matrix
RegDM = horzcat(NormMat(1:end,1:MinIndex-1),NormMat(1:end,MinIndex+1:end));

%Cross Validation
%Division of Data Sets
[DMSize,DMAttrib]=size(RegDM);
SetSize = fix(DMSize/3);
Set2Size = DMSize - (SetSize * 2);
RandIndex = randperm(DMSize);

DM1 = RegDM(RandIndex(1:SetSize),:);
DM1X = DM1(:,1:(DMAttrib-1));
DM1Y = DM1(:,DMAttrib:DMAttrib);

DM2 = RegDM((Set2Size+1):(SetSize * 2),:);
DM2X = DM2(:,1:(DMAttrib-1));
DM2Y = DM2(:,DMAttrib:DMAttrib);

DM3 = RegDM(((SetSize * 2)+1):DMSize,:);
DM3X = DM3(:,1:(DMAttrib-1));
DM3Y = DM3(:,DMAttrib:DMAttrib);



%Degree 1
%K=1
W = zeros(3,1);
TrSetK1X = DM1X;
ValSetK1X = [DM2X;DM3X];
TrSetK1Y = DM1Y;
ValSetK1Y = [DM2Y;DM3Y];
D1W1=gradientDescent(TrSetK1X,TrSetK1Y , W, alpha, iterations);
D1C1 = computeCost(ValSetK1X, ValSetK1Y, D1W1);

%K=2
TrSetK2X = DM2X;
ValSetK2X = [DM1X;DM3X];
TrSetK2Y = DM2Y;
ValSetK2Y = [DM1Y;DM3Y];

D1W2=gradientDescent(TrSetK2X,TrSetK2Y , W, alpha, iterations);
D1C2 = computeCost(ValSetK2X, ValSetK2Y, D1W2);


%K=3
TrSetK3X = DM3X;
ValSetK3X = [DM1X;DM2X];
TrSetK3Y = DM3Y;
ValSetK3Y = [DM1Y;DM2Y];
D1W3 =gradientDescent(TrSetK3X,TrSetK3Y , W, alpha, iterations);
D1C3 = computeCost(ValSetK3X, ValSetK3Y, D1W3);

D1MeanCost = mean([D1C1,D1C2,D1C3])

%Degree 2
D2M1X=DMSQ(DM1X);
D2M2X=DMSQ(DM2X);
D2M3X=DMSQ(DM3X);

%K=1

W = zeros(5,1);
TrSetK1X = D2M1X;
ValSetK1X = [D2M2X;D2M3X];
TrSetK1Y = DM1Y;
ValSetK1Y = [DM2Y;DM3Y];
D2W1=gradientDescent(TrSetK1X,TrSetK1Y , W, alpha, iterations);
D2C1 = computeCost(ValSetK1X, ValSetK1Y, D2W1);

%K=2
TrSetK2X = D2M2X;
ValSetK2X = [D2M1X;D2M3X];
TrSetK2Y = DM2Y;
ValSetK2Y = [DM1Y;DM3Y];
D2W2=gradientDescent(TrSetK2X,TrSetK2Y , W, alpha, iterations);
D2C2 = computeCost(ValSetK2X, ValSetK2Y, D2W2);


%K=3
TrSetK3X = D2M3X;
ValSetK3X = [D2M1X;D2M2X];
TrSetK3Y = DM3Y;
ValSetK3Y = [DM1Y;DM2Y];
D2W3 =gradientDescent(TrSetK3X,TrSetK3Y , W, alpha, iterations);
D2C3 = computeCost(ValSetK3X, ValSetK3Y, D2W3);

D2MeanCost = mean([D2C1,D2C2,D2C3])

%Iteration part
IterDSet = RegDM;
Iter = 100;
if D2MeanCost < D1MeanCost
    IterDSetX = IterDSet(:,1:(DMAttrib-1));
    IterDSetY = IterDSet(:,DMAttrib:DMAttrib);
    IterDSetX=DMSQ(IterDSetX);
    IterDSet = [IterDSetX,IterDSetY];
end

[DMSize,DMAttrib]=size(IterDSet);
TrDSize = fix(DMSize/2);
TstDSize = DMSize - TrDSize;
MErr = [];
GErr = [];
Wa = [];
for i=1:Iter
    %Data Randomisation
    RandIndex = randperm(DMSize);
    TrDMat = IterDSet(RandIndex(1:TrDSize),:);
    TstDMat = IterDSet((TrDSize+1):DMSize,:);
    TrDMatX = TrDMat(:,1:(DMAttrib-1));
    TrDMatY = TrDMat(:,DMAttrib:DMAttrib);
    TstDMatX = TstDMat(:,1:(DMAttrib-1));
    TstDMatY = TstDMat(:,DMAttrib:DMAttrib);
    Wsize = size(TrDMatX,2);
    W = zeros(Wsize,1);
    
    %W and Error Calculations
    W =gradientDescent(TrDMatX,TrDMatY , W, alpha, iterations);
    Wa= [Wa,W];
    MErr = [MErr,computeCost(TrDMatX, TrDMatY, W)];
    GErr = [GErr,computeCost(TstDMatX, TstDMatY, W)];
end

%Spike Removal
 GErr(GErr > 1) = median(GErr);
 MErr(GErr > 1) = median(MErr);

%Graph Plot
figure
plot(MErr);
hold on;
plot(GErr);
legend('Modelling Errors','Generalization Errors');
title('Modelling Errors and Generalization Errors Vs Iteration');
xlabel('Iterations');
ylabel('Errors');
hold off;

%Min-Max Calculations
MinModError = min(MErr)
MaxModError = max(MErr)
AvgModError = mean(MErr)
MinGenError = min(GErr)
MaxGenError = max(GErr)
AvgGenError = mean(GErr)


%Contour
W0_vals = linspace(-10, 30, 100);
W1_vals = linspace(-10, 20, 100);

[DMSize,DMAttrib]=size(IterDSet);
DataX = IterDSet(:,1:(DMAttrib-1));
DataY= IterDSet(:,DMAttrib:DMAttrib);

% % initialize J_vals to a matrix of 0's
J_vals = zeros(length(W0_vals), length(W1_vals));

 % Fill out J_vals
for i = 1:length(W0_vals)
    for j = 1:length(W1_vals)
	  t = [W(1); W0_vals(i);  W1_vals(j);W(4);W(5)];    
	  J_vals(i,j) = computeCost(DataX,DataY, t);
    end
end

J_vals = J_vals';
figure;
subplot(1,2,1)
surf(W0_vals, W1_vals, J_vals)
    xlabel('w_1'); ylabel('w_2');
    title('Surface')

% Contour plot
subplot(1,2,2)

% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(W0_vals, W1_vals, J_vals, logspace(-5, 30, 150))
xlabel('W_1'); ylabel('W_2');
hold on;
plot(W(2), W(3), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title('Contour')

figure
plot(DataX(:,2), DataX*W,'o')
legend('X1 Feature', '','Linear Regression')
title('Linear Regression for X1, Degree=2')

hold on;
xi=-1:0.01:4;
yi=interp1(DataX(:,2), DataX*W,xi,'spline');
plot(xi,yi)
hold off;

figure
plot(DataX(:,3), DataX*W,'o')
legend('X2 Feature', 'Linear Regression')
title('Linear Regression for X2, Degree=2')
hold on;
xi=-1:0.01:5;
yi=interp1(DataX(:,3), DataX*W,xi,'spline');
plot(xi,yi)
hold off;

figure
plot(DataX(:,4), DataX*W,'o')
legend('X2 Feature', 'Linear Regression')
title('Linear Regression for X2, Degree=2')
hold on;
xi=-1:0.01:15;
yi=interp1(DataX(:,4), DataX*W,xi,'linear');
plot(xi,yi)
hold off;

figure
plot(DataX(:,5), DataX*W,'o');
legend('X2 Feature', 'Linear Regression')
title('Linear Regression for X2, Degree=2')
hold on;
xi=-3:0.01:18;
yi=interp1(DataX(:,5), DataX*W,xi,'linear');
plot(xi,yi,'-')
hold off


