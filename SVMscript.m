%% NEJM SPRINT Data Analysis: Master Script
% Senan Ebrahim, Aditya Kalluri, Mark Kalinich
% February 1, 2017

% Load NEJM SPRINT data
load('alldata.mat')
load('bigevents.mat')
load('bigsafety.mat')

simpleSVM = SVMshotgun(alldata, bigevents);

safepredSVM = SVMshotgun(alldata,bigsafety);

% safeSVM = SVMshotgun(safety,bigevents);

% inter = [];
% for i = 1:size(alldata,2)
%     for j = i:size(alldata,2)
%         inter = [inter alldata(:,i).*alldata(:,j)];
%     end
% end
% 
% including interaction variables
% allwinter = [alldata inter];
% interSVM = SVMshotgun(allwinter, bigevents);