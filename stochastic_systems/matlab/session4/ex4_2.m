close all
clear

rng(42);

N = 2000;

X1 = normrnd(0,1,N);
X2 = normrnd(0,1,N);

Y = X1 + X2;
histogram(X1)
hold on
histogram(Y)

%%
close all
clear

rng(42);

N = 2000;

X1 = unifrnd(-1*sqrt(3),sqrt(3),N);
X2 = unifrnd(-1*sqrt(3),sqrt(3),N);

Y = X1 + X2;
histogram(X1)
hold on
histogram(Y)

