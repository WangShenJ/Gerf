clc;clear;close all;
load('../../logs/s4-exp1/debug/ray.mat');
load('../../logs/s4-exp1/debug/gateway1_pos.mat');

r_o = rays(1:100:end,1:3);
r_d = rays(1:100:end,4:6)*10;
r_d = r_d + r_o;
xx = [r_o(:,1)'; r_d(:,1)'];
yy = [r_o(:,2)'; r_d(:,2)'];
zz = [r_o(:,3)'; r_d(:,3)'];

% xx = [zeros(1,324); r_d(:,1)'];
% yy = [zeros(1,324); r_d(:,2)'];
% zz = [zeros(1,324); r_d(:,3)'];


radius = 20;

%%


%%
figure;
hold on;
line(xx,yy,zz);

scatter3(pos(1,:), pos(2,:), pos(3,:),'MarkerFaceColor','r');

t = linspace(0,2*pi,360);
r = linspace(0,radius,90);
[T, R] = meshgrid(t, r);
[X, Y] = pol2cart(T,R);
Z = sqrt(radius.^2 - (X.^2 + Y.^2));
surf(X,Y,real(Z), zeros(90,360),'FaceAlpha',0.5);
shading interp;
daspect([1,1,1]);
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');
