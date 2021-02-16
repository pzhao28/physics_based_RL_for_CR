a = [-6.25 -3.5 -4.25 -4.75 -4 -5 -2 -2.25 -1 -0.5 -0.25 -0.5 -0.25 -0.5 -0.75 -0.5 -0.5 -0.25 -0.25 -0.5 -0.5 0 -1 -0.25 -0.75];
b = [-4.5 -3.25 -2.5 -0.5 -0.75 -0.75 0 -0.25 -0.25 -0.25 0 0 0 -0.5 -0.25 -0.25 0 -0.5 -0.25 0 0 0 0 -0.25 -0.25];

x = 1:length(a);

plot(x*20,-a,'-ok',x*20,-b,'-or')
xlabel('episode')
ylabel('average conflicts per two flight hours')
legend('classical method', 'SSD based method')

