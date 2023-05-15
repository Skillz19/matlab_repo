classdef main
    properties
        shape = [224 224];
        net_type = Enum_Models.RESNET50;
        learn_rate = 0.001;
        optimizer = 'sgdm';
        epoch = 20;
        batch_size = 32;
        val_patience = 25;
        result_txt = 'results.txt';
        display = false;
    end
    methods
        function run(self)
            %load the dataset
            dl = LoadData(self.shape,true, ...
                'Chest_xray');
            [train_data, ~] = dl.load_data(Enum_Loadtype.ENUM_TRAINING);
            
            % Display a selection of the images
            % minibatch = read(train_data);
            % imshow(imtile(minibatch.input))
            % display image shape
            
            % preview data
            % batch = preview(train_data);
            % image_shape = batch{1,1};
            % disp(image_shape)
            
            % load validation data
            [val_data, ~] = dl.load_data(Enum_Loadtype.ENUM_VALIDATION);
            
            % show sample of loaded validation data
            % minibatch = read(val_data);
            % imshow(imtile(minibatch.input))
            
            % load transfer learning model
            lm = LoadModel(self.net_type,self.learn_rate);
            model = lm.load_model();
            
            % train network
            trained_net = self.train(train_data, val_data, model);
            
            %load test data
            [test_data, test_labels] = dl.load_data(Enum_Loadtype.ENUM_TESTING);
            
            % Test model performance
            result = self.evaluate(test_data, test_labels, trained_net);
            
            % save model performance 
            self.save_result(result);
            
        end
        % test trained model
        function test(self, trained_net)
            dl = LoadData(self.shape,true,'Chest_xray');
            %load test data
            [test_data, test_labels] = dl.load_data(Enum_Loadtype.ENUM_TESTING);
            
            % Test model performance
            result = self.evaluate(test_data, test_labels, trained_net);

            % save model performance 
            self.save_result(result);
        end
    end
    methods(Access=private)
        function trained_net = train(self, data, val_data, model)
            % configure training option
            opts = trainingOptions(self.optimizer,"InitialLearnRate",self.learn_rate, ...
                "MaxEpochs",self.epoch,"MiniBatchSize",self.batch_size,"ValidationData",val_data ...
                ,"ValidationPatience",self.val_patience,"LearnRateDropPeriod",10,"Plots","training-progress");
            
            % Train the network
            trained_net = trainNetwork(data,model,opts);
            
            % save the model
            self.save_model(trained_net);
        end
        function save_model(self, trained_net)
            save(['Models\' char(self.net_type) '.mat'],"trained_net")
        end
        
        % evaluate the network
        function result = evaluate(~, data, labels, trained_net)
            predictedLabels = classify(trained_net, data); 
            accuracy = mean(predictedLabels == labels);
            
            confMat = confusionmat(labels, predictedLabels);
            numClasses = size(confMat, 1);
            precision = zeros(numClasses, 1);
            recall = zeros(numClasses, 1);
            for k = 1:numClasses
                precision(k) = confMat(k,k) / sum(confMat(k,:));
                recall(k) = confMat(k,k) / sum(confMat(:,k));
                
            end
            F1 = 2 * (precision .* recall) ./ (precision + recall);
            result = ['Accuracy: ', num2str(mean(accuracy)), ' F1 score: ', ...
                num2str(mean(F1)), ' Recall: ', num2str(mean(recall)), ...
                ' Precision ', num2str(mean(precision))];
            % disp results
            if self.display
                disp(F1)
                disp(recall)
                disp(precision)
                disp(confMat)
                disp(result)
            end
        end
        
        % save formatted string for result
        function save_result(self, txt)
            fid = fopen(self.result_txt,"a");
            if fid > -1
                fprintf(fid,'Test results for %s: %s \n',char(self.net_type), txt);
                fclose(fid);
            end
        end
    end
end