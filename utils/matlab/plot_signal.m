clc;clear;close all;
load("../../data/s23-exp3/signals.mat");
gateway1 = gateway1(1:100,:,:);
amp = gateway1(:,1,1);
phs = gateway1(:,1,2);
figure;


% plot(s_real,s_imag);
plot(amp*100);


