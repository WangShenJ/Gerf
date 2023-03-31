clc;clear;close all;
% load('../data/pq504_05m_0622.mat');


radius = 2;

%%
alpha = 266 / 180 *pi;
beta = 45 / 180*pi;
y = radius * cos(alpha) * cos(beta);
z = radius * sin(beta);
x = radius * sin(alpha) * cos(beta);
l1 = [0.24, 0.24, 0.24, 0.24,...
      0.08, 0.08, 0.08, 0.08,...
      -0.08, -0.08, -0.08, -0.08,...
      -0.24, -0.24, -0.24, -0.24];

l2 = [-0.24, -0.08, 0.08, 0.24,-0.24, -0.08, 0.08, 0.24,-0.24, -0.08, 0.08, 0.24,-0.24, -0.08, 0.08, 0.24];

ey = y + l1*sin(beta)*cos(alpha) + l2*sin(alpha);
ez = z - l1*cos(beta);
ex = x + l1*sin(beta)*sin(alpha) - l2*cos(alpha);

%%
figure;
hold on;
scatter3(x,y,z,'MarkerFaceColor','b')
scatter3(ex,ey,ez,'MarkerFaceColor','r');


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
