dl = LoadData([224 224],true,'Chest_xray');
[test_data, test_labels] = dl.load_data(Enum_Loadtype.ENUM_TESTING);
            % Test model performance
            predictedLabels = classify(trained_net, test_data); 
            accuracy = mean(predictedLabels == test_labels);
            disp(accuracy)

            confMat = confusionmat(test_labels, predictedLabels);
            numClasses = size(confMat, 1);
            precision = zeros(numClasses, 1);
            recall = zeros(numClasses, 1);
            for k = 1:numClasses
                precision(k) = confMat(k,k) / sum(confMat(:,k));
                recall(k) = confMat(k,k) / sum(confMat(k,:));
            end
            F1 = 2 * (precision .* recall) ./ (precision + recall);
            result = ['Accuracy: ', num2str(mean(accuracy)), ' F1 score: ', ...
                num2str(mean(F1)), ' Recall: ', num2str(mean(recall)), ...
                ' Precision ', num2str(mean(precision))];
            disp(result)
            % save result to file
            