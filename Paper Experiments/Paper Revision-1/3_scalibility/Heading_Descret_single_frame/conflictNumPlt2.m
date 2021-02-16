num_CR=load('conflict_num_CR.csv');
num_noCR=load('conflict_num_noCR.csv');
num_vb = load('conflict_num_vb.csv');
num_vb = num_vb/10;
x = 1:19;
plot(x+2,num_noCR,x+2,num_vb,x+2,num_CR);