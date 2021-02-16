clc;
clear;
load('aircrafts0.mat')
load('aircrafts1.mat')
load('aircrafts2.mat')
ac00 = [ac0,ac1,ac2,ac3,ac4,ac5];
subplot(2,3,4)
for i=1:5
hold on;
fill([ac00(i).vector(1),ac00(i).T1(1),ac00(i).T2(1)]+40,[ac00(i).vector(2),ac00(i).T1(2),ac00(i).T2(2)]+40,'r','LineStyle','none');
hold off
end
xlim([0,80])
ylim([0,80])
xlabel('nautical mile')
ylabel('nautical mile')
axis('square')
box on
subplot(2,3,5)
ac10 = [ac100,ac101,ac102];
for i=1:3
hold on;
fill([ac10(i).vector(1),ac10(i).T1(1),ac10(i).T2(1)]+40,[ac10(i).vector(2),ac10(i).T1(2),ac10(i).T2(2)]+40,'r','LineStyle','none');
hold off
end
xlim([0,80])
ylim([0,80])
xlabel('nautical mile')
axis('square')
box on
subplot(2,3,6)
ac20 = [ac200,ac201,ac202,ac203,ac204];
for i=1:5
hold on;
fill([ac20(i).vector(1)+40,ac20(i).T1(1),ac20(i).T2(1)],[ac20(i).vector(2)+40,ac20(i).T1(2),ac20(i).T2(2)],'r','LineStyle','none');
hold off
end
xlim([0,80])
ylim([0,80])
xlabel('nautical mile')
axis('square')
box on


subplot(2,3,1)
for i=1:5
hold on;
viscircles([ac00(i).x ac00(i).y],5,'Color','r');
quiver(ac00(i).x,ac00(i).y,ac00(i).vector(1),ac00(i).vector(2),0);
hold off
end
xlim([0,80]);
ylim([0,80]);
xlabel('nautical mile')
ylabel('nautical mile')
axis('square');
box on

subplot(2,3,2)
for i=1:3
hold on;
viscircles([ac10(i).x ac10(i).y],5,'Color','r');
quiver(ac10(i).x,ac10(i).y,ac10(i).vector(1),ac10(i).vector(2),0);
hold off
end
xlim([0,80]);
ylim([0,80]);
xlabel('nautical mile')
axis('square');
box on

subplot(2,3,3)
for i=1:5
hold on;
viscircles([ac20(i).x ac20(i).y],5,'Color','r');
quiver(ac20(i).x,ac20(i).y,ac20(i).vector(1),ac20(i).vector(2),0);
hold off
end
xlim([0,80]);
ylim([0,80]);
xlabel('nautical mile')
axis('square');
box on

