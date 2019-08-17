%% Dream Challenge

% This script process data for the pregancy outcome prediction from the
% dream challenge. 

% =========================== Overview ==========================
% A basic need in pregnancy care is to establish gestational age, and
% inaccurate estimates may lead to unnecessary interventions and
% sub-optimal patient management. Current approaches to establish
% gestational age rely on patient’s recollection of her last menstrual
% period and/or ultrasound, with the latter being not only costly but also
% less accurate if not performed during the first trimester of pregnancy.
% Therefore development of an inexpensive and accurate molecular clock of
% pregnancy would be of benefit to patients and health care systems.
% Participants in sub-challenge 1 (Prediction of gestational age) will be
% given whole blood gene expression data collected from pregnant women to
% develop prediction models for the gestational age at blood draw. 

% Another challenge in obstetrics, in both low and high-income countries,
% is identification and treatment of women at risk of developing the ‘great
% obstetrical syndromes‘. Of these, preterm birth (PTB), defined as giving
% birth prior to completion of 37 weeks of gestation, is the leading cause
% of newborn deaths and long-term complications including motor, cognitive,
% and behavioral impairment. Participants in sub-challenge 2 (Prediction of
% preterm birth) will be given whole blood gene expression data collected
% from pregnant women to develop prediction models to determine the risk
% preterm birth.

% make a fresh start
clear; close all force; clc;

% change the directory 
% cd('/Users/sinkala/Documents/MATLAB/DreamPregrancy')

%% Load the data 

fprintf('\n Loading the pre-processing the expression data \n')

% load the rnaSeq data and the associated clinical data for the patients
% outcomes
rnaSeq = readtable('Gene Expression.csv');
rnaSeq.Properties.VariableNames(1) = "HugoSymbol" ;
outcomes = readtable('Train and Test IDs.csv');

% get the training and test data
trainingClinicals = outcomes(outcomes.Train == 1, :) ;
testClinicals = outcomes(outcomes.Train == 0, :) ;

%% Normalise the RNAseq data

% get the unnormalised gene counts
counts = rnaSeq{:,2:end} ;

% estimate pseudo-reference with geometric mean row by row
pseudoRefSample = geomean(counts,2);
nz = pseudoRefSample > 0;
ratios = bsxfun(@rdivide,counts(nz,:),pseudoRefSample(nz));
sizeFactors = median(ratios,1);

% transform to common scale
normCounts = bsxfun(@rdivide,counts,sizeFactors);
normCounts(1:10,:);

% You can appreciate the effect of this normalization by using the function
% boxplot to represent statistical measures such as median, quartiles,
% minimum and maximum.

figure;
subplot(2,1,1)
maboxplot( log2(counts(:,1:10)) ,'title','Raw Read Count',...
    'orientation','horizontal')
ylabel('sample')
xlabel('log2(counts)')

subplot(2,1,2)
maboxplot( log2(normCounts(:,1:10)),'title','Normalized Read Count',...
    'orientation','horizontal')
ylabel('sample')
xlabel('log2(counts)')

% ======== add the normalised counts to the brca table =========
rnaSeq{:,2:end} = normCounts ;

clear counts normCounts sizeFactors nz ratios pseudoRefSample ...
    landMarkGenes 

%% Remove the non variable genes

% obtain expression measurements
expression = rnaSeq{:,2:end};
samples = rnaSeq.Properties.VariableNames(2:end) ;
genes = rnaSeq.HugoSymbol; % obtain the genes

% remove nan values
nanIndices = any(isnan(expression),2);
expression(nanIndices,:) = [];
genes(nanIndices) = [];
numel(genes)

% ======================== Filter Out Genes =====================

% Gene profiling experiments typically include genes that exhibit little
% variation in their profile and are generally not of interest. These genes
% are commonly removed from the data.

% Mask = genevarfilter(Data) calculates the variance for each gene
% expression profile in Data and returns Mask, which identifies the gene
% expression profiles with a variance less than the 10th percentile. Mask
% is a logical vector with one element for each row in Data. The elements
% of Mask corresponding to rows with a variance greater than the threshold
% have a value of 1, and those with a variance less than the threshold are
% 0.

for times = 1:4 % 4
    mask = genevarfilter(expression);
    
    expression = expression(mask,:);
    genes = genes(mask);
    numel(genes)
    
    % filter out genes below a certain fold change threshold
    [~,expression,genes] = ...
        genelowvalfilter(expression,genes,'absval',log2(2));
    numel(genes)
    
    % filter genes below a certain percentile: VERY POWERFUL discriminant
    [~,expression,genes] = ...
        geneentropyfilter(expression,genes,'prctile',20);
    numel(genes)
end

% finally convert back to a table 
rnaSeq = [genes, array2table(expression,'VariableNames', ...
    samples) ] ;
rnaSeq.Properties.VariableNames(1) = "HugoSymbol" ;

clear outcomes mask expression nanIndices times

%% Get the training and test data

% transpose the table before getting the features 
rnaSeq = rows2vars(rnaSeq,'VariableNamesSource','HugoSymbol');
rnaSeq.Properties.VariableNames(1) = "SampleID" ;

% get the training data and test data and convert to zscores
trainingData = rnaSeq( ...
    ismember(rnaSeq.SampleID,trainingClinicals.SampleID), :) ;
trainingData{:,2:end} = zscore(trainingData{:,2:end} );
testData = rnaSeq( ...
    ismember(rnaSeq.SampleID,testClinicals.SampleID), :) ;
testData{:,2:end} = zscore(testData{:,2:end} ) ;

% throw in an assertions
assert(all(ismember(trainingClinicals.SampleID ,trainingData.SampleID)))
assert(all(ismember(testClinicals.SampleID ,testData.SampleID)))

%% Visaulise for any batch effect 

fprintf('\n running PCA and K-means Clustering \n') 

% arrange the test clinical data in the same way as the rnaSeq data
[~,locThese] = ismember(trainingData.SampleID, trainingClinicals.SampleID );
trainingClinicals = trainingClinicals(locThese, :) ;

% throw in an assertion 
assert(all(strcmp(trainingClinicals.SampleID ,trainingData.SampleID)))

expressionT = trainingData{:,2:end}' ;

% apply kmean clustering 
idxBest = kmeans(trainingData{:,2:end},3,'Distance',...
    'sqeuclidean','Display','final','Replicates',500);

% set the color of the trimisters and the also conver the trimister data to
% categorical data 
colorsTrimister = [0.9448 0.3377 0.1112 ; ...
    0.5909 0.8001 0.6803 ; 0.4893 0.3692 0.3897] ;
trimisters = str2double(trainingClinicals.GA) ;
trimisters(trimisters < 13) = 1 ;
trimisters(trimisters >= 25) = 3 ;
trimisters(trimisters > 4) = 2 ; % 4 because the values have been converted

% perform a PCA for the RNAseq data
[coeff,~,~,~,explained] = pca(expressionT);

% make a 2D plot
figure(); clf
set(gcf,'position',[100,50,800,800]);
subplot(2,2,1)
gscatter(coeff(:,1),coeff(:,2),trimisters',colorsTrimister,'..',40)
set(gca,'LineWidth',1.5,'FontSize',14,'FontWeight','bold')
title('Expression in Pregancy','FontSize',14)
legend({'1 Trim','2 Trim','3Trim'})
xlabel(['PC1-(',num2str(round(explained(1))),'%)'] );
ylabel(['PC2-(',num2str(round(explained(2))),'%)'] );

% groups color set here
grpColor = [0.3658 0.5329 0.6963; 0.7635 0.9727 0.0938 ; ...
        0.6279 0.1920 0.5254] ;

% make a 2D plot
subplot(2,2,2)
gscatter(coeff(:,1),coeff(:,2),idxBest',grpColor,'..',40)
set(gca,'LineWidth',1.5,'FontSize',14,'FontWeight','bold')
title('Expression in Pregancy: Groups','FontSize',14)
legend({'group1','group2','group3'})
xlabel(['PC1-(',num2str(round(explained(1))),'%)'] );
ylabel(['PC2-(',num2str(round(explained(2))),'%)'] );

% groups color set here
batchColor = rand(numel(unique(trainingClinicals.Batch)), 3) ;

subplot(2,2,3)
gscatter(coeff(:,1),coeff(:,2),trainingClinicals.Batch, batchColor,'..',40)
set(gca,'LineWidth',1.5,'FontSize',14,'FontWeight','bold')
title('Expression in Pregancy: Batch Effect','FontSize',14)
xlabel(['PC1-(',num2str(round(explained(1))),'%)'] );
ylabel(['PC2-(',num2str(round(explained(2))),'%)'] );

clear  grpColor trimister latent explained colorsTrimister ...
    idxBest expressionT locThese batchColor

%% Extact features using lasso and NCA for Regression 

% combine the two table together and save a copy to focus on everything
mrnaFeatures = innerjoin( trainingData, trainingClinicals(:,1:2)) ;

% convert to double and change the variable names
mrnaFeatures.(width(mrnaFeatures)) = ...
    str2double(mrnaFeatures.(width(mrnaFeatures) )  );
mrnaFeatures.Properties.VariableNames(end) = "GestationalAge" ;
mrnaFeatures = mrnaFeatures(:,[2:end]);

% convert the gestationAge to log for faster training 
mrnaFeatures.GestationalAge = mrnaFeatures.GestationalAge ;

% extract features using lasso
fprintf('\n Extracting features using Lasso \n') 
mrnaFeatures = feature_selection_lasso(mrnaFeatures) ;

% % extract feature using NCA
% fprintf('\n Extracting features using NCA for regression \n')
% mrnaFeatures = feature_selection_nca_robust(mrnaFeatures ) ;

%% Train a Gaussian Process Regression Model 

fprintf('\n Training Gaussian Process Regression Model \n')
% get the training and test data 
% get the test and training samples and the validation samples
obs = mrnaFeatures{:,1:end-1} ;
grp = mrnaFeatures{:,end};

% ==========  Divide data into training and test sets ================
% Use |cvpartition| to divide data into a training set of size 160 and
% a test set of size 56. Both the training set and the test set have
% roughly the same group proportions as in |grp|.
rng(10) % For reproducibility
cvp = cvpartition(grp,'holdout',0.2)  ; % samples to hold out
xTrain = obs(cvp.training,:);
yTrain = grp(cvp.training,:) ;
xTest  = obs(cvp.test,:);
yTest  = grp(cvp.test,:)  ;

% Train a regression model
rng default
GPRmodel = fitrgp(xTrain,yTrain,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',60,'UseParallel',true) );

% predict using the regression model 
YPred = predict(GPRmodel ,xTest);
YPred(YPred < 8) = 10;
YPred(YPred > 42) = 40;

% Evaluate the performance of the model by calculating: The percentage of
% predictions within an acceptable error margin The root-mean-square error
% (RMSE) of the predicted and actual angles of rotation Calculate the
% prediction error between the predicted and actual angles of rotation.
predictionError = yTest - YPred ;

% Use the root-mean-square error (RMSE) to measure the differences between
% the predicted and actual angles of rotation.
rmse = sqrt(mean(predictionError.^2)) ;

%% Use the Trained SVM Model in the learner App make predictions

% ========================== SUBMITTED MODEL ==========================
% Your submission has been scored!{'rmse': 5.4169} 

% This model was save. Therefore, I simply load the model here. Alternative
% the model may be retrained using the function provided in the
% trainSVMR_Final.m provided with other supporting code
load('bestSVMmodel.mat')

xTable = mrnaFeatures(cvp.test,1:end-1) ;
YPred2 = trainedModelSVM.predictFcn(xTable) ;

% predict using the regression model 
YPred(YPred < 8) = 9;
YPred(YPred > 42) = 40;

% Evaluate the performance of the model by calculating: The percentage of
% predictions within an acceptable error margin The root-mean-square error
% (RMSE) of the predicted and actual angles of rotation Calculate the
% prediction error between the predicted and actual angles of rotation.

predictionError2 = yTest - YPred2 ;

% Use the root-mean-square error (RMSE) to measure the differences between
% the predicted and actual angles of rotation.

rmse2 = sqrt(mean(predictionError2.^2)) 

% % predict using the regression model and change the values for those
% above 42 and below 8 weeks
myPredictions = trainedModelSVM.predictFcn( pregTestData );
myPredictions(myPredictions < 8) = 9;
myPredictions(myPredictions > 42) = 40;

% make a table and save it to excel 
myPredictions = addvars(testData(:,1),myPredictions,...
    'NewVariableNames','GA') ;

writetable(myPredictions,'sinkalaPredictionsSVMLeaner.csv')

%% Train an Ensemble Model to check the Error

% Since the GPR and SVM model were found to be the most accurate. Let's try
% to make ensemble model of these and see how well they perform

% ========================== SUBMITTED MODEL ==========================
% Your submission has been scored!{'rmse': 5.9493}

fprintf('\n Training Ensemble Super Methods Hyperparameter Optimization\n')

rng(0) % for reproducibility 
superGPR = @(x,y)fitrgp(x,y, 'KernelFunction','ardsquaredexponential',...
    'Verbose',1,'Optimizer','quasinewton','OptimizerOptions',opts, ...
    'KernelParameters',[sigmaL0;sigmaF0],'Sigma',sigmaN0,...
    'InitialStepSize','auto');

superSVM = @(x,y)fitrsvm(x,y,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'UseParallel',true));

superGPR_relational =  @(x,y)fitrgp(x,y,...
    'KernelFunction','RationalQuadratic',...
    'Verbose',1,'Optimizer','lbfgs','OptimizerOptions',opts, ...
    'KernelParameters',[30;1;sigmaF0],'Sigma',sigmaN0,...
    'InitialStepSize','auto');

% Initialize Ensemble
ens = custom_ensemble;
ens.learners = {superSVM, superGPR , superGPR_relational};
% ens.meta_learner = superGPR ; % this implies that stacking is used
ens.meta_learner = {}; % this implies that mean is used

% Train Ensemble
ens = ens.fit(xTrain, yTrain);

% Predict
yFitHyper = ens.predict(xTest);
yFitHyper(yFitHyper < 8) = 9;
yFitHyper(yFitHyper> 42) = 40 ;
ensembleErrorHyper = yTest - yFitHyper ;
ensembleRMSEHyper = sqrt(mean(ensembleErrorHyper.^2)) 

% Now train the model on all the data 
ens = ens.fit(mrnaFeatures{:,1:end-1}, mrnaFeatures.GestationalAge);

% make predictions for the data 
myPredictions = ens.predict( pregTestData{:,:} );
myPredictions(myPredictions < 8) = 9;
myPredictions(myPredictions > 42) = 40;

% make a table and save it to excel 
myPredictions = addvars(testData(:,1),myPredictions,...
    'NewVariableNames','GA') ;

writetable(myPredictions,'sinkalaPredictionsSuper.csv')

%%