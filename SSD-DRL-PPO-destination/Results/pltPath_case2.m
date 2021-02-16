mov = xlsread("test_moving_2.csv");
A = tril(ones(length(mov)));
path_x = A * mov(:,1);
path_y = A * mov(:,2);

seq = 1:length(mov);

drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),8,'k' );

intruders_start = [[-40 40.0]; [40 40/sqrt(3)]; [40 40.0]; [40 0.0]];
intruders_end = [[0 40.0]; [0 80/sqrt(3)]; [-40 40.0]; [-40 40.0]];

intruder_path_x = zeros(4, length(mov));
intruder_path_y = zeros(4, length(mov));
    hold on;

for i=1:4
    intruder_start = intruders_start(i,:) + [path_x(i), path_y(i)];
    intruder_end = intruders_end(i,:) + [path_x(i), path_y(i)];
    intruder_sin_theta = (intruder_end(2) - intruder_start(2))/sqrt(sum((intruder_start - intruder_end).^2));
    intruder_cos_theta = (intruder_end(1) - intruder_start(1))/sqrt(sum((intruder_start - intruder_end).^2));
    intruder_path_x(i,:) = intruder_start(1)*ones(1,length(mov)) +  4 * intruder_cos_theta * seq;
    intruder_path_y(i,:) = intruder_start(2)*ones(1,length(mov)) +  4 * intruder_sin_theta * seq;
    plot(intruder_path_x(i,:), intruder_path_y(i,:), 'k');
    
    %drawArrow([intruder_path_x(i,2) intruder_path_x(i,4)], [intruder_path_y(i,2) intruder_path_y(i,4)]);
        
end
line([path_x(12) 0], [path_y(12) 140],'Color','b','LineStyle','-')
intruder = plot(intruder_path_x(i,:), intruder_path_y(i,:), 'k', 'DisplayName','intruder paths');
legend(intruder, 'intruder paths');
plt_plan = line([0 0],[0 140],'Color','red','LineStyle','-.');
plt_path = plot(path_x(1:12), path_y(1:12),'b');

xlim([-40,40]);
hold off;
    
% path_own = animatedline('Color', 'k', 'Linestyle', '-');
% path_intruder1 = animatedline('Color', 'r', 'Linestyle', '-');
% path_intruder2 = animatedline('Color', 'r', 'Linestyle', '-');
% set(gca, 'XLim', [-40 40], 'YLim', [0 800]);
% intruder1_start = [-20 80];
% intruder1_end = [20 0];
% intruder1_sin_theta = (intruder1_end(2) - intruder1_start(2))/sqrt(sum((intruder1_start - intruder1_end).^2));
% intruder1_cos_theta = (intruder1_end(1) - intruder1_start(1))/sqrt(sum((intruder1_start - intruder1_end).^2));
% intruder1_vel = 3.5;
% 
% intruder2_start = [20 80];
% intruder2_end = [-20 0];
% intruder2_sin_theta = (intruder2_end(2) - intruder2_start(2))/sqrt(sum((intruder2_start - intruder2_end).^2));
% intruder2_cos_theta = (intruder2_end(1) - intruder2_start(1))/sqrt(sum((intruder2_start - intruder2_end).^2));
% intruder2_vel = 3.8;
% 
% intruder1_x = intruder1_start(1);
% intruder1_y = intruder1_start(2)-40;
% 
% intruder2_x = path_x(100) + 20;
% intruder2_y = path_y(100) + 40;
% 
% filename = 'test_2ac.gif';
% 
% for i=1:length(path_x)
%     addpoints(path_own, path_x(i), path_y(i));
%     intruder1_x = intruder1_x + intruder1_cos_theta * intruder1_vel;
%     intruder1_y = intruder1_y + intruder1_sin_theta * intruder1_vel;
%     addpoints(path_intruder1, intruder1_x, intruder1_y)
%     if i>= 100
%         intruder2_x = intruder2_x + intruder2_cos_theta * intruder2_vel;
%         intruder2_y = intruder2_y + intruder2_sin_theta * intruder2_vel;
%         addpoints(path_intruder2, intruder2_x, intruder2_y)
%     end
%     
%     drawnow
%     %pause(0.01)
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     if i == 1
%       imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
%     else
%       imwrite(imind,cm,filename,'gif','DelayTime',0.05, 'WriteMode','append');
%     end
% end
% 
