clear all;clearvars -global;close all;clc;

fid1 = fopen('ac1_data.txt');
ac1 = textscan(fid1,'%f%f%f');
fid1 = fclose(fid1);

fid2 = fopen('ac2_data.txt');
ac2 = textscan(fid2,'%f%f%f');
fid2 = fclose(fid2);

fid3 = fopen('ac3_data.txt');
ac3 = textscan(fid3,'%f%f%f');
fid1 = fclose(fid3);

D=trajectory3([ac1{1},ac2{1},ac3{1}],[ac1{2},ac2{2},ac3{2}],[ac1{3},ac2{3},ac3{3}]*20-2000,zeros(200,3),zeros(200,3),[zeros(200,1) -0.5*ones(200,1) 0.5*ones(200,1)],0.06,1,'747');

filename1 = 'LRACAS_demo.gif';
for i = 1 :3: 200
    im = frame2im(D(i)); 
      [imind,cm] = rgb2ind(im,512);
      if i == 1 
          imwrite(imind,cm,filename1,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename1,'gif','delaytime',0,'WriteMode','append'); 
      end 
end