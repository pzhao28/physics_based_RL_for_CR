mov = xlsread("test_case2_10ac.csv");
A = tril(ones(length(mov)));
path_x = A * mov(:,1);
path_y = A * mov(:,2);
path_own = animatedline('Color', 'k', 'Linestyle', '-');
path_intruder1 = animatedline('Color', 'r', 'Linestyle', '-');
path_intruder2 = animatedline('Color', 'r', 'Linestyle', '-');
set(gca, 'XLim', [-40 40], 'YLim', [0 800]);
intruder1_start = [20.99147241274844, 0.0];
intruder1_end = [22.78831399024993, 12.976075642772633];
intruder1_sin_theta = (intruder1_end(2) - intruder1_start(2))/sqrt(sum((intruder1_start - intruder1_end).^2));
intruder1_cos_theta = (intruder1_end(1) - intruder1_start(1))/sqrt(sum((intruder1_start - intruder1_end).^2));
intruder1_vel = 3.5;

intruder2_start = [40.490848291596706, 0.0];
intruder2_end = [-20 0];
intruder2_sin_theta = (intruder2_end(2) - intruder2_start(2))/sqrt(sum((intruder2_start - intruder2_end).^2));
intruder2_cos_theta = (intruder2_end(1) - intruder2_start(1))/sqrt(sum((intruder2_start - intruder2_end).^2));
intruder2_vel = 3.8;

intruder1_x = intruder1_start(1);
intruder1_y = intruder1_start(2);

intruder2_x = path_x(100) + 20;
intruder2_y = path_y(100) +80;

filename = 'test_2ac.gif';

for i=1:length(path_x)
    addpoints(path_own, path_x(i), path_y(i));
    intruder1_x = intruder1_x + intruder1_cos_theta * intruder1_vel;
    intruder1_y = intruder1_y + intruder1_sin_theta * intruder1_vel;
    addpoints(path_intruder1, intruder1_x, intruder1_y)
    if i>= 100
        intruder2_x = intruder2_x + intruder2_cos_theta * intruder2_vel;
        intruder2_y = intruder2_y + intruder2_sin_theta * intruder2_vel;
        addpoints(path_intruder2, intruder2_x, intruder2_y)
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

