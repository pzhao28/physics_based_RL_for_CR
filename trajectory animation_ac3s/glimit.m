function dmin=glimit(x)
DIM=7;
a1=x(1:DIM);
a2=x(DIM+1:end);
[t1,f1]=FLIGHTver2(a1);
[t2,f2]=FLIGHTver2(a2);
dd=(f1(:,1:2)-f2(:,1:2));
d1=(sum(dd.^2,2)).^0.5;
d2=abs(f1(:,3)-f2(:,3));
dmin=min(d1);
end