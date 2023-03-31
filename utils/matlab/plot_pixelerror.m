clc;clear;close all;
load('../../logs/s23-exp4/pixel_error.mat');

figure;
set(gcf,'unit', 'inches', 'position', [5,5,9,4]);  % 设置图片长宽
axes('linewidth', 2, 'box', 'on', 'FontSize', 16); % 设置坐标轴线宽

hold on;
cdfplot(err);