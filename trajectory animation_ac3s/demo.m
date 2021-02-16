clear all;clearvars -global;close all;clc;
N=2e3;
dof=7;
I=0;
mean1=[260,40,10,0,0,10,-10];
cov1=0.05;
var1=abs(mean1*cov1);
mean2=[300,0,0,42000,-2000,10,-10];
cov2=0.05;
var2=abs(mean2*cov2);

for i=1:dof
    Sam1(:,i)=normrnd(mean1(i),var1(i),N,1);
end
for i=1:dof
    Sam2(:,i)=normrnd(mean2(i),var2(i),N,1);
end
Sam=[Sam1,Sam2];
% figure
% plot(t,f(:,2),'r-');
% hold on
% plot(t,f(:,1),'*');
for i=1:N
%dmin=glimit(Sam(i,:));

[t1,f1]=FLIGHTver2(Sam1(i,:));
[t2,f2]=FLIGHTver2(Sam2(i,:));
dd=(f1(:,1:2)-f2(:,1:2));
d1=(sum(dd.^2,2)).^0.5;
d2=abs(f1(:,3)-f2(:,3));
dmin=min(d1);

if dmin<9260
    I=I+1;
end
end
% figure
% plot(t1,f1(:,2)),'b-';
% hold on
% plot(t,f1(:,1),'*');
% figure
% plot(f1(:,1),f1(:,2),'r');
% hold on 
% plot(f2(:,1),f2(:,2),'g');
% figure
% plot(t1,f1(:,3),'r');
% hold on
% plot(t2,f2(:,3),'g');
% [t1,f1]=FLIGHTver2(mean1);
% [t2,f2]=FLIGHTver2(mean2);
% dd=(f1(:,1:2)-f2(:,1:2));
% d1=(sum(dd.^2,2)).^0.5;
% d2=abs(f1(:,3)-f2(:,3));
% dmin=min(d1);
% figure 
% plot(t1,d1);
