clear all;
mov = xlsread("test_moving.csv");
A = tril(ones(length(mov)));
path_x = A * mov(:,1);
path_y = A * mov(:,2);
path_own = animatedline('Color', 'k', 'Linestyle', '-');
path_intruder = animatedline('Color', 'r', 'Linestyle', '-');
set(gca, 'XLim', [-40 40], 'YLim', [0 300]);

vels = [3.5,  3.8, 4.7, 4.7];
starts = [[0, 80.0]; [40, 80.0]; [0, 40.0]; [80, 40.0]];
ends = [[80, 40.0]; [40, 0.0]; [80, 80.0]; [0, 80.0]];
steps = tril(ones(length(vels)))*[4, 32, 24, 13]';


intruder_start = zeros(length(vels),2);
intruder_end = zeros(length(vels),2);
intruder_sin_theta = zeros(length(vels),1);
intruder_cos_theta = zeros(length(vels),1);
intruder_vel = zeros(length(vels),1);
for i=1:length(vels)
    intruder_start(i,:) = starts(i,:);
    intruder_end(i,:) = ends(i,:);
    intruder_sin_theta(i) = (intruder_end(i,2) - intruder_start(i,2))/sqrt(sum((intruder_start(i,:) - intruder_end(i,:)).^2));
    intruder_cos_theta(i) = (intruder_end(i,1) - intruder_start(i,1))/sqrt(sum((intruder_start(i,:) - intruder_end(i,:)).^2));
    intruder_vel(i) = vels(i);
    path_intruder(i) = animatedline('Color', 'r', 'Linestyle', '-');
end

intruder_x = zeros(length(vels),1);
intruder_y = zeros(length(vels),1);
intruder_x(1) = path_x(1) + intruder_start(1,1)-40;
intruder_y(1) = path_y(1) + intruder_start(1,2)-20;
filename = 'test_10ac.gif';
j = 1;
for i=1:length(path_x)
    addpoints(path_own, path_x(i), path_y(i));
    if i== steps(j)&&j<length(vels)
        j = j + 1;
        intruder_x(j) = path_x(i) + intruder_start(j,1) - 40;
        intruder_y(j) = path_y(i) + intruder_start(j,2) - 40;
        if j == 2
            intruder_y(j) = intruder_y(i)+80;
        end
    end
    for k = 1:j
        
        intruder_x(k) = intruder_x(k) + intruder_cos_theta(k) * intruder_vel(k);
        intruder_y(k) = intruder_y(k) + intruder_sin_theta(k) * intruder_vel(k);
        addpoints(path_intruder(k), intruder_x(k), intruder_y(k))
    end
    
    drawnow
    %pause(0.01)
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
      imwrite(imind,cm,filename,'gif','DelayTime',0.05, 'WriteMode','append');
    end
end

