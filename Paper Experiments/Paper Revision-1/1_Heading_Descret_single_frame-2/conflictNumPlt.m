a = xlsread('data_1.csv');
%b = xlsread('data_2.csv');
%c = xlsread('data_3.csv');
d = xlsread('data_4.csv');
x = 1:length(a);
plot(x,-a,x,-d)
