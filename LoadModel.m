classdef LoadModel
    %LOADMODEL build network based on transfer learning
    %   load and edit pretrained models for binary classification
    
    properties
        m_networktype
        m_learningrate
    end
    
    methods
        function self = LoadModel(networktype,learningrate)
            %LOADMODEL Construct an instance of this class
            self.m_networktype = networktype;
            self.m_learningrate = learningrate;
        end
        
        function network = load_model(self)
            %METHOD1 build and return specified model
            if self.m_networktype == Enum_Models.RESNET50
                network = self.build_resnet50();
            elseif self.m_networktype == Enum_Models.EFFICIENTNET
                network = self.build_efficientnet();
            elseif self.m_networktype == Enum_Models.DENSENET
                network = self.build_densenet();
            else
                error('Unknown network model specified!')
            end
        end
    end
    %define private methods
    methods (Access = private)
        function network = build_resnet50(self)
            network = resnet50;
            network = self.replace_layers(network);
            disp(network.Layers)
        end
        function network = build_efficientnet(self)
            network = efficientnetb0;
            network.Layers
            network = self.replace_layers(network);
            disp(network.Layers)
        end
        function network = build_densenet(self)
            network = densenet201;
            network = self.replace_layers(network);
            disp(network.Layers)
        end
        % replace fully connected layer and output classification layers
        % with binary classification layers
        function network = replace_layers(~,net)
            network = layerGraph(net);
            len = size(network.Layers);
            fc2_layer = fullyConnectedLayer(2,'Name','fc_2');
            tf_output = classificationLayer('Name','tf_output');
            % replace fully connected layer
            network = replaceLayer(network,network.Layers(len-2:len-2).Name,fc2_layer);
            % replace the last output layer
            network = replaceLayer(network,network.Layers(len:len).Name,tf_output);
        end
    end
end

