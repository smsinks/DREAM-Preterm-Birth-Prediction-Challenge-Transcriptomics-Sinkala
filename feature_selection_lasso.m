%This function select feature for machine learning using lasso elasltic net

function [genetic_train] = feature_selection_lasso(geneticData_drug)

% INPUT:
% geneticData: a table containing genetic data with the drug on the end
% column.

% OUTPUT:
% genetic_train: training data with specific features selected

% load(fullfile(matlabroot,'examples','stats','robotarm.mat'))

data = geneticData_drug;

% this data is in tabular format need to get Xtrain and ytrain and the
% other data Xtest and ytest at 20%
X_store = data(:,2:end-1);
y_store = data(:,end);

XTrain = table2array( data(:,2:end-1));
yTrain = table2array( data(:,end));

% The alpha value changes lasso to elastic net which can select more
% feature than the number of samples.
alphas = [0.1:0.1:0.8] ;
% Find the coefficients of a regularized linear regression model using
% 10-fold cross-validation and the elastic net method with Alpha = 0.75.
% Use the largest Lambda value such that the mean squared error (MSE) is
% within one standard error of the minimum MSE.

for kk = 1:length(alphas)
    cur_alpha = alphas(1,kk) ;
    [B,FitInfo] = lasso(XTrain,yTrain,'Alpha',cur_alpha,'CV',10);
    MSE = FitInfo.MSE(FitInfo.Index1SE) ;
    if kk == 1
        idxLambda1SE = FitInfo.Index1SE; 
        mseBestSoFar = MSE;
        Best_beta = B ;
    else
        if mseBestSoFar > MSE 
            mseBestSoFar = MSE ;
            idxLambda1SE = FitInfo.Index1SE ; 
            Best_beta = B ;
        end
    end
end

% Find the coefficients of a regularized linear regression model using
% 10-fold cross-validation and the elastic net method with Alpha = 0.75.
% Use the largest Lambda value such that the mean squared error (MSE) is
% within one standard error of the minimum MSE.
% 
% [B, FitInfo] = lasso(XTrain,yTrain,'Alpha',0.7,'CV',10);
% idxLambda1SE = FitInfo.IndexMinMSE;

% [B,FitInfo] = lassoglm(X,y,'poisson','Alpha',1.0,'CV',10);
% [B,FitInfo] = lassoglm(X,y,'Alpha',1.0,'CV',10) ;

%Examine the cross-validation plot to see the effect of the Lambda
%regularization parameter.

% figure(77)
% lassoPlot(B,FitInfo,'plottype','CV');
% legend('show'); % Show legend

% Find the nonzero model coefficients corresponding to the two identified
% points. Sometimes, there are no coefficents that are nonezero; in such
% cases the original input table is returned.
fprintf('\n Elastic Net Best Lambda1SE is at %d \n', idxLambda1SE);
size(B)
size(Best_beta)
size(X_store)
selected =  Best_beta(:,idxLambda1SE)~=0;
selectedFeatures = X_store(:,selected) ;
genetic_train = [data(:,1), selectedFeatures, y_store] ;

end
