clc;clear;close all;
load('../../logs/s23-exp2/debug/gateway1_pos.mat');
tagpos = readmatrix("../../data/s23-exp2/tagpos.txt");
testset_ind = readmatrix("../../data/s23-exp2/test.txt");
trainset_ind = readmatrix("../../data/s23-exp2/train.txt");

%%
testset = tagpos(testset_ind,:);
trainset = tagpos(trainset_ind, :);

figure;
% set(gcf,'unit', 'inches', 'position', [5,5,9,4]);  % 设置图片长宽
axes('linewidth', 2, 'FontSize', 16); % 设置坐标轴线宽


hold on;
htr = scatter3(trainset(:,1),trainset(:,2),trainset(:,3),"MarkerFaceColor","blue","SizeData",36);
hts = scatter3(testset(:,1),testset(:,2),testset(:,3),"MarkerFaceColor","red","SizeData",36);
legend("train set", "test set");

%% Env
% scatter3(pos(1,:), pos(2,:), pos(3,:),'MarkerFaceColor','r');
% 
% radius = 5
% t = linspace(0,2*pi,360);
% r = linspace(0,radius,90);
% [T, R] = meshgrid(t, r);
% [X, Y] = pol2cart(T,R);
% Z = sqrt(radius.^2 - (X.^2 + Y.^2));
% surf(X,Y,real(Z), zeros(90,360),'FaceAlpha',0.5);
% shading interp;
% daspect([1,1,1]);
% xlabel('x-axis');
% ylabel('y-axis');
% zlabel('z-axis');