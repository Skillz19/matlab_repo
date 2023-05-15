classdef LoadData
    properties
        % tuple specifying image shape
        m_shape
        % (bool) convert images to rgb
        m_gray2rgb
        % (string) parent directory
        m_parent_dir
        m_train_dir
        m_val_dir
        m_test_dir
        % rotation 
        m_rand_rotaion = [-90 90]
    end

    methods
        function self = LoadData(shape, gray2rgb, ...
                parent_dir)
            self.m_shape = shape;
            self.m_gray2rgb = gray2rgb;
            self.m_parent_dir = parent_dir;
            self.m_train_dir = strcat(parent_dir, '\train');
            self.m_test_dir = strcat(parent_dir, '\test');
            self.m_val_dir = strcat(parent_dir, '\val');
        end
        % load image data
        function [data, labels] = load_data(self, load_type)
            if load_type == Enum_Loadtype.ENUM_TRAINING
                data = self.load_data_from_folder(self.m_train_dir);
            elseif load_type == Enum_Loadtype.ENUM_VALIDATION
                data = self.load_data_from_folder(self.m_val_dir);
            elseif load_type == Enum_Loadtype.ENUM_TESTING
                data = self.load_data_from_folder(self.m_test_dir);
            else 
                error(['Unknown load type enumeration specified! Option' ...
                    'can either be ENUM_TRAINING | ENUM_VALIDATION |' ...
                    ' ENUM_TESTING']);
            end
            labels = data.Labels;
            data = self.add_preprocessing(data);
        end
    end
    % private class methods
    methods (Access = private)
        % add preprocess function to dataset
        function auimds = add_preprocessing(self, data)
            data.ReadFcn = @LoadData.read_function;
            %specify options for data augmentation
            imageAugmenter = imageDataAugmenter('RandRotation',self.m_rand_rotaion, ...
                'RandXReflection',true);
            %define augmented data store
            auimds = augmentedImageDatastore(self.m_shape,data, ...
                "ColorPreprocessing","gray2rgb","DataAugmentation",imageAugmenter);
        end
        % define function for reading imageDataStore object 
    end
    % static private methods
    methods (Static, Access=private)
        function data = read_function(filename)
            % function for reading image
            data = imread(filename);
            
        end
        % load data from a specified folder
        function folder_data = load_data_from_folder(directory)
            folder_data = imageDatastore(directory,'IncludeSubFolders',true ...
                ,'LabelSource','foldernames');
            folder_data = shuffle(folder_data);
        end
        
    end
end








