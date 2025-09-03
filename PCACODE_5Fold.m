% --- NEW CODE STARTS HERE ---
% Step 8: Prepare Data for Machine Learning
% We will use the PCA scores as features for classification.
numPCs = 10;
X = scores(:, 1:numPCs); % Feature matrix [60 samples x 10 features]

% Create label vector - simpler approach
Y_labels = repelem({'U0', 'U5', 'G0', 'G5'}, 15)'; % 60x1 cell array
Y_categorical = categorical(Y_labels);
classNames = categories(Y_categorical);
numClasses = length(classNames);

% Convert to numeric labels for easier handling
Y_numeric = grp2idx(Y_categorical);

% Step 9: Define the 5-Fold Cross-Validation Setup
rng(42); % Set seed for reproducibility
k = 5;

% Create cross-validation partition using numeric labels
cv = cvpartition(Y_numeric, 'KFold', k, 'Stratify', true);

% Step 10: Initialize Classifiers to Evaluate
classifiers = {
    'LDA', @(x,y) fitcdiscr(x, y, 'DiscrimType', 'linear');
    'KNN', @(x,y) fitcknn(x, y, 'NumNeighbors', 3);
    'Naive Bayes', @(x,y) fitcnb(x, y);
    'SVM', @(x,y) fitcecoc(x, y);
    'Decision Tree', @(x,y) fitctree(x, y);
    'Random Forest', @(x,y) fitcensemble(x, y, 'Method', 'Bag', 'NumLearningCycles', 50);
    'RUSBoost', @(x,y) fitcensemble(x, y, 'Method', 'RUSBoost', 'NumLearningCycles', 50);

};

numClassifiers = size(classifiers, 1);

% Preallocate results storage
results = struct();
for c = 1:numClassifiers
    results(c).Accuracy = zeros(k, 1);
    results(c).Precision = zeros(k, 1);
    results(c).Recall = zeros(k, 1);
    results(c).F1 = zeros(k, 1);
    results(c).MSE = zeros(k, 1);
end

% Step 11: Perform 5-Fold Cross-Validation
fprintf('Starting 5-Fold Cross-Validation...\n');

for fold = 1:k
    fprintf('\nProcessing Fold %d/%d...\n', fold, k);
    
    % Get training and test indices
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Extract data using numeric indices (not logical)
    XTrain = X(trainIdx, :);
    XTest = X(testIdx, :);
    YTrain = Y_numeric(trainIdx);
    YTest = Y_numeric(testIdx);
    
    % Convert back to categorical for confusion matrix
    YTest_categorical = Y_categorical(testIdx);
    
    % Train and evaluate each classifier
    for c = 1:numClassifiers
        classifierName = classifiers{c, 1};
        classifierFunc = classifiers{c, 2};
        
        fprintf('  Training %s...', classifierName);
        
        try
            % Train the model
            model = classifierFunc(XTrain, YTrain);
            
            % Predict on test set
            YPred_numeric = predict(model, XTest);
            
            % Convert predictions to categorical for confusion matrix
            YPred_categorical = categorical(YPred_numeric, 1:numClasses, classNames);
            
            % Calculate confusion matrix
            C = confusionmat(YTest_categorical, YPred_categorical);
            
            % Calculate performance metrics
            accuracy = sum(diag(C)) / sum(C, 'all');
            
            % Calculate class-wise metrics
            precisionPerClass = zeros(numClasses, 1);
            recallPerClass = zeros(numClasses, 1);
            f1PerClass = zeros(numClasses, 1);
            
            for j = 1:numClasses
                TP = C(j, j);
                FP = sum(C(:, j)) - TP;
                FN = sum(C(j, :)) - TP;
                
                precisionPerClass(j) = TP / (TP + FP + eps);
                recallPerClass(j) = TP / (TP + FN + eps);
                f1PerClass(j) = 2 * (precisionPerClass(j) * recallPerClass(j)) / ...
                                (precisionPerClass(j) + recallPerClass(j) + eps);
            end
            
            % Macro-average metrics
            precision = mean(precisionPerClass);
            recall = mean(recallPerClass);
            f1 = mean(f1PerClass);
            
            % Calculate MSE
            mse = mean((YTest - YPred_numeric).^2);
            
            % Store results
            results(c).Accuracy(fold) = accuracy;
            results(c).Precision(fold) = precision;
            results(c).Recall(fold) = recall;
            results(c).F1(fold) = f1;
            results(c).MSE(fold) = mse;
            
            fprintf(' Done (Accuracy: %.3f)\n', accuracy);
            
        catch ME
            warning('Error training %s: %s', classifierName, ME.message);
            results(c).Accuracy(fold) = NaN;
            results(c).Precision(fold) = NaN;
            results(c).Recall(fold) = NaN;
            results(c).F1(fold) = NaN;
            results(c).MSE(fold) = NaN;
            fprintf(' Error\n');
        end
    end
end

% ... [End of your cross-validation loop] ...

% --- DIAGNOSTIC CODE FOR BOOSTED TREES ---
fprintf('\n=== DIAGNOSTIC CHECK ===\n');
for c = 1:numClassifiers
    classifierName = classifiers{c, 1};
    if any(isnan(results(c).Accuracy))
        fprintf('❌ Classifier %s failed with NaN results. Check for errors.\n', classifierName);
        % Check which folds failed
        failedFolds = find(isnan(results(c).Accuracy));
        fprintf('   Failed in folds: %s\n', mat2str(failedFolds));
    else
        fprintf('✅ %s: All folds completed successfully\n', classifierName);
    end
end

% --- VALIDATION CHECKS FOR PERFECT SCORES ---
fprintf('\n=== DATA VALIDATION ===\n');
fprintf('Total samples: %d\n', size(X, 1));
fprintf('Class distribution:\n');

% Get class counts and percentages
classCounts = histcounts(Y_numeric, 1:numClasses+1);
for i = 1:numClasses
    percentage = (classCounts(i) / size(X, 1)) * 100;
    fprintf('  %s: %d samples (%.1f%%)\n', classNames{i}, classCounts(i), percentage);
end

% Check if any class has very few samples
if any(classCounts < 5)
    fprintf('⚠️  Warning: Some classes have very few samples (<5)\n');
end

fprintf('Feature matrix size: %d samples x %d features\n', size(X, 1), size(X, 2));

% ... [Continue with Step 12: Calculate Mean and Standard Deviation] ...


% Step 12: Calculate Mean and Standard Deviation
fprintf('\n=== FINAL RESULTS (5-Fold Cross-Validation) ===\n');

% Create results table
resultData = cell(numClassifiers, 9);
for c = 1:numClassifiers
    classifierName = classifiers{c, 1};
    
    meanAccuracy = mean(results(c).Accuracy, 'omitnan');
    stdAccuracy = std(results(c).Accuracy, 'omitnan');
    
    meanPrecision = mean(results(c).Precision, 'omitnan');
    stdPrecision = std(results(c).Precision, 'omitnan');
    
    meanRecall = mean(results(c).Recall, 'omitnan');
    stdRecall = std(results(c).Recall, 'omitnan');
    
    meanF1 = mean(results(c).F1, 'omitnan');
    stdF1 = std(results(c).F1, 'omitnan');
    
    meanMSE = mean(results(c).MSE, 'omitnan');
    stdMSE = std(results(c).MSE, 'omitnan');
    
    % Store in cell array
    resultData{c, 1} = classifierName;
    resultData{c, 2} = meanAccuracy;
    resultData{c, 3} = stdAccuracy;
    resultData{c, 4} = meanPrecision;
    resultData{c, 5} = stdPrecision;
    resultData{c, 6} = meanRecall;
    resultData{c, 7} = stdRecall;
    resultData{c, 8} = meanF1;
    resultData{c, 9} = stdF1;
    resultData{c, 10} = meanMSE;
    resultData{c, 11} = stdMSE;
    
    % Display results
    fprintf('\n%s:\n', classifierName);
    fprintf('  Accuracy:  %.3f (±%.3f)\n', meanAccuracy, stdAccuracy);
    fprintf('  Precision: %.3f (±%.3f)\n', meanPrecision, stdPrecision);
    fprintf('  Recall:    %.3f (±%.3f)\n', meanRecall, stdRecall);
    fprintf('  F1-Score:  %.3f (±%.3f)\n', meanF1, stdF1);
    fprintf('  MSE:       %.3f (±%.3f)\n', meanMSE, stdMSE);
end

% Convert to table
resultsTable = cell2table(resultData, 'VariableNames', {
    'Classifier', 'Accuracy', 'Accuracy_STD', 'Precision', 'Precision_STD', ...
    'Recall', 'Recall_STD', 'F1_Score', 'F1_Score_STD', 'MSE', 'MSE_STD'
});

% Step 13: Create Visualization
figure('Position', [100, 100, 1200, 500]);

% Subplot 1: Accuracy Comparison
subplot(1, 2, 1);
accuracies = cell2mat(resultData(:, 2));
accuracies_std = cell2mat(resultData(:, 3));
classifierNamesShort = {'LDA', 'KNN', 'NB', 'SVM', 'DT', 'RF', 'Boost'};

bar(accuracies, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
errorbar(1:numClassifiers, accuracies, accuracies_std, 'k.', 'LineWidth', 1.5);
set(gca, 'XTick', 1:numClassifiers, 'XTickLabel', classifierNamesShort);
ylabel('Mean Accuracy');
title('Classifier Performance (5-Fold CV)');
ylim([0, 1.1]);
grid on;

% Add value labels
for i = 1:numClassifiers
    text(i, accuracies(i) + 0.02, sprintf('%.3f', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Step 14: Save Results
writetable(resultsTable, 'Classifier_Performance_Results.xlsx');
fprintf('\nResults saved to Classifier_Performance_Results.xlsx\n');


% ... [All your existing code until the end] ...

% --- ADDITIONAL DIAGNOSTICS (add at the very end) ---
function runAdditionalDiagnostics(results, classifiers, X, Y_numeric, classNames, numClasses)
    fprintf('\n\n=== ADDITIONAL DIAGNOSTICS ===\n');
    
    % Boosted Trees diagnostic
    for c = 1:length(classifiers)
        if any(isnan(results(c).Accuracy))
            fprintf('❌ %s failed during training\n', classifiers{c, 1});
        end
    end
    
    % Data validation
    fprintf('Dataset Info:\n');
    fprintf('  Total samples: %d\n', size(X, 1));
    fprintf('  Features: %d\n', size(X, 2));
    fprintf('  Classes: %d\n', numClasses);
    
    classCounts = histcounts(Y_numeric, 1:numClasses+1);
    for i = 1:numClasses
        fprintf('  %s: %d samples\n', classNames{i}, classCounts(i));
    end
end

% Run the diagnostics
runAdditionalDiagnostics(results, classifiers, X, Y_numeric, classNames, numClasses);