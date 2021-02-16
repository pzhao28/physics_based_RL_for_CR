a = xlsread("data_no_destination.xlsx");
b = xlsread("data_no_action_penalty.csv");
c = xlsread("data_action_penalty_2.csv");

x = 1:42;

plot(x*20,a(x),'.-k', x*20, b(x), '.-r',x*20, c, '.-b')
xlabel('episodes')
ylabel('average reward')
%ylabel('average conflict per 5-days continuous flight')
legend('no destination, no action penalty', 'destination, no action penalty', 'destination, action penalty')

