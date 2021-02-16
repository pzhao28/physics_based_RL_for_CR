a = xlsread("average_conflict.xlsx");

x = 1:length(a);

plot(x*20,-a,'-ok')
xlabel('episodes')
ylabel('average conflict per 5-days continuous flight')
%legend('method 1')

