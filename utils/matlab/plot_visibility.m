clc;clear;close all;
load("../../logs/s31-exp2/visibility/points.mat");
load("../../logs/s31-exp2/visibility/gamma.mat");
load("../../logs/s31-exp2/debug/gateway1_pos.mat");

%%
pts = pts(1:6:end,:);
alpha = alpha(1:6:end, :);
ind = find(pts(:,3)<1.1);
pts = pts(ind,:);
alpha = alpha(ind,:);

%%
% pts = reshape(pts,[32,360,90,3]);
% pts = pts(1:32,:,1:35,:);
% pts = reshape(pts,[],3);
% 
% alpha = reshape(alpha,[32,360,90,1]);
% alpha = alpha(1:32,:,1:35,:);
% alpha = reshape(alpha,[],1);
% pts = pts(1:6:end,:);
% alpha = alpha(1:6:end, :);
%%

blue = [zeros(length(alpha),1),zeros(length(alpha),1),ones(length(alpha),1)];

h = scatter3(pts(:,1), pts(:,2), pts(:,3),[],alpha,"filled","AlphaData",alpha,"MarkerEdgeAlpha","flat","MarkerFaceAlpha","flat")
hold on;
scatter3(pos(1,:), pos(2,:), pos(3,:),'MarkerFaceColor','r');
% N = h.MarkerHandle.FaceColorData;
% N(4,:) = alpha;
% set(h.MarkerHandle,"FaceColorData", N);
% 
% 
% radius = 5;

% figure;

% for i=1:1000
%     i
%     s = scatter3(temp(i,1), temp(i,2), temp(i,3),[],'b',"filled");
%     s.MarkerFaceAlpha = alpha(i);
%     hold on;
% end