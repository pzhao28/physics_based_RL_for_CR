load('pos_mem.mat')
x1 = pos_mem(:,1,1);
y1 = pos_mem(:,1,2);
x2 = pos_mem(:,2,1);
y2 = pos_mem(:,2,2);
x3 = pos_mem(:,3,1);
y3 = pos_mem(:,3,2);
plot(x1,y1,'b',x2,y2,'g',x3,y3,'r')