clear all;clearvars -global;close all;clc;
N=2e3;
dof=7;
I=0;
thre=9260;
mean1=[260,40,10,0,0,10,-10];
cov1=0.05;
var1=abs(mean1*cov1);
mean2=[300,0,0,40000,-7000,10,-10];
cov2=0.05;
var2=abs(mean2*cov2);
[t1,f1]=FLIGHTver2(mean1);
[t2,f2]=FLIGHTver2(mean2);
dd=(f1(:,1:2)-f2(:,1:2));
d1=(sum(dd.^2,2)).^0.5;
d2=abs(f1(:,3)-f2(:,3));
dmin=min(d1);
% f1=ff1(1:300,:);
% f2=ff2(1:300,:);
numSteps = length(t1);
% numSteps = 50;
% numSteps = 100;
% Calculate the trajectory by linear interpolation of the x and y coordinate.
trajectory1=f1(:,1:2)';
trajectory2=f2(:,1:2)';
% Make figure with axes.
i=0:0.01:2*pi;
% figure;
% axis square; 
% hold on;
% %axis equal;
% 
% % Make the frames for the movie.
% filename = 'testdemo.gif';
% for frameNr = 1 : numSteps
%      cla;
%      set(gca,'XLim',[0 14e4], 'YLim', [-7000 18e4]);
% title('horizotal flight path of the aircrafts');
% xlabel('East/m');
% ylabel('North/m');
% % 	axis manual
% 	% Plot the point.
% 	plot( trajectory1(1,frameNr), trajectory1(2,frameNr), 'r*' );
%     
%     
%     plot( trajectory2(1,frameNr), trajectory2(2,frameNr), 'g*' );
%     legend('Aircraft#1','Aircraft#2');
%     plot(thre*cos(i)+trajectory1(1,frameNr),thre*sin(i)+trajectory1(2,frameNr),'r-');
%     plot(thre*cos(i)+trajectory2(1,frameNr),thre*sin(i)+trajectory2(2,frameNr),'g-');
% % 	plot(thre*cos(i)+trajectory2(1,frameNr),thre*sin(i)+trajectory2(1,frameNr),'g-');
% 	% Plot the path line.
%     plot(f1(1:frameNr,1),f1(1:frameNr,2),'r-');
%     plot(f2(1:frameNr,1),f2(1:frameNr,2),'g-');
%     drawnow;
% % 	line( 'XData', [trajectory(1,1) trajectory(1,frameNr)], ...
% %           'YData', [trajectory(2,1) trajectory(2,frameNr)] );
%       
%     
% 	
% 	% Get the frame for the animation.
% 	frames(frameNr) = getframe;
%     
% %     im = frame2im(frames(frameNr)); 
% %       [imind,cm] = rgb2ind(im,256);
% %     if frameNr == 1 
% %           imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
% %       else 
% %           imwrite(imind,cm,filename,'gif','WriteMode','append'); 
% %       end 
% end
% %  movie(frames);
% for frameNr = 1 :3: numSteps
%     im = frame2im(frames(frameNr)); 
%       [imind,cm] = rgb2ind(im,256);
%       if frameNr == 1 
%           imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
%       else 
%           imwrite(imind,cm,filename,'gif','delaytime',0,'WriteMode','append'); 
%       end 
% end
% h = figure;
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = 'testAnimated.gif';
% for n = 1:0.5:5
%     % Draw plot for y = x.^n
%     x = 0:0.01:1;
%     y = x.^n;
%     plot(x,y)
%    
%     plot(x,y.^-1)
%     drawnow 
%       % Capture the plot as an image 
%       frame = getframe(h); 
%       im = frame2im(frame); 
%       [imind,cm] = rgb2ind(im,256); 
%       % Write to the GIF File 
%       if n == 1 
%           imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
%       else 
%           imwrite(imind,cm,filename,'gif','WriteMode','append'); 
%       end 
%   end
% varargin=cell(1,1);
% varargin(1)={'747'};
% D=trajectory3(f1(:,1),f1(:,2),-f1(:,3),f1(:,4),f1(:,5),f1(:,6),0.0001,10,'747');
% 
% 
% 
D=trajectory3([f1(:,1),f2(:,1)],[f1(:,2),f2(:,2)],[-f1(:,3),-f2(:,3)],[f1(:,4),f2(:,4)],[f1(:,5),f2(:,5)],[f1(:,6),f2(:,6)],0.0001,1,'747');

filename1 = '3D_demo.gif';
for i = 1 :3: 1000
    im = frame2im(D(i)); 
      [imind,cm] = rgb2ind(im,512);
      if i == 1 
          imwrite(imind,cm,filename1,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename1,'gif','delaytime',0,'WriteMode','append'); 
      end 
end
