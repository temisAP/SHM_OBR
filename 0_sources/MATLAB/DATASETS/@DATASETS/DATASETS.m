classdef DATASETS < handle

    %% Properties

    properties
        path
    end

    %% Constructor

    methods (Access = public)

        function obj = DATASETS()
            welcome_message(obj);
        end

    end
    
    %% Métodos privados

    methods (Access = private)

        function welcome_message(obj)
            disp('***LOADING DATASET***');
        end

    end
    
    %% Métodos públicos
    
    methods (Access = private)

        function welcome_message2(obj)
            disp('***LOADING DATASET***');
        end

    end

    

end
