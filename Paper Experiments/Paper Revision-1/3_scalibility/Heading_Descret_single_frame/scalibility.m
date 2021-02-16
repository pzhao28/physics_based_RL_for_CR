y1 = csvread('data_N1.csv');
y2 = csvread('data_N3.csv');
y3 = csvread('data_N5.csv');
y4 = csvread('data_N7.csv');
y5 = csvread('data_N9.csv');
y  = csvread('data_N_rand.csv');

x = 1:500;
plot(x,-y1(x),'.-',x,-y2(x),'.-',x,-y3(x),'.-',x,-y4(x),'.-',x,-y5(x),'.-');
hold on;
plot(x,-y(x),'.--b','LineWidth',1)
legend('number of intruders = 1','number of intruders = 3','number of intruders = 5','number of intruders = 7','number of intruders = 9','random number of intruders')
xlabel('episode')
ylabel('number of conflicts per two flight hours')
