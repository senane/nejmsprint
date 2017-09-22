%% NEJM SPRINT Data Analysis: SVM Function
% Senan Ebrahim, Aditya Kalluri, Mark Kalinich
% February 1, 2017

% This function creates SVM models for all event outcomes given 
% baseline characteristics as input data. 

% Outputs the AUC and class-loss under 10-fold CV for each variable.

% Data supposed to have been cleaned of NaN but ensuring

function [results] = SVMshotgun(data, events)

results=nan(2,size(events,2));

allnonan = data(~any(isnan(data),2),:);

for i=1:size(events,2)
    
    event = events(:,i);
    evnonan = event(~any(isnan(data),2),:);
    
    
% Downsampling data is not necessary but is illustrated here

%     dsfactor = int(length(evnonan)/sum(evnonan));
    
%     ev1 = evnonan(evnonan==0,:);
%     ev2 = evnonan(evnonan==1,:);
%     ev1 = ev1(1:length(ev2),:);
%     
%     dat1 = allnonan(evnonan==0,:);
%     dat2 = allnonan(evnonan==1,:);
%     dat1 = dat1(1:length(dat2),:);
    

%     trainev = [ev1(trainInd); ev2(trainInd)];
%     traindat = [dat1(trainInd,:); dat2(trainInd,:)];
%     
%     testev = [ev1(testInd); ev2(testInd)];
%     testdat = [dat1(testInd,:); dat2(testInd,:)];


    
    [trainInd,valInd,testInd] = dividerand(length(evnonan),0.8,0,0.2);
    
    % 80-20 train-test split
    trainev = evnonan(trainInd);
    traindat = allnonan(trainInd,:);
    
    testev = evnonan(testInd);
    testdat = allnonan(testInd,:);
    
    assignin('base','testdat',testdat)
    assignin('base','testev',testev)
    assignin('base','traindat',traindat)
    assignin('base','trainev',trainev)
    
    mdl = fitcsvm(traindat,trainev,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
    % mdl = fitPosterior(mdl);
    % [~,score] = resubPredict(mdl);
    %[X, Y, T, AUC] = perfcurve(logical(trainev),score(:,logical(mdl.ClassNames)),'true');
   
    [label,score] = predict(mdl,testdat);
    [X, Y, T, AUC] = perfcurve(logical(testev),score(:,logical(mdl.ClassNames)),'true');
    results(1,i) = AUC;
    
    cvmdl = crossval(mdl);
    classloss = kfoldLoss(cvmdl);
    results(2,i) = classloss;
end

end