function TL_processRecordingWrapper_v2(pathname , varargin)
%% FUNCTION: Takes a folder 'YYMMDD' that contains all experimental files. 
% Conducts all necessary steps from arranging the files to sorting the
% spikes for in vivo recordings using igor, tdt, and webcam software

% INPUTS:
% [pathname] - pathname to YYMMDD folder. this folder must contain csv
% files from igor and tdt, and ddt files from tdt
% [varargin] - pathname to output from tdt conversions to ddt and csv, in
% case you want the code to be triggered by full conversion on the tdt
% computer
% NOTES:
% Must remember to fill out excel spreadsheet containing experimental
% information. This spreadsheet should be in the same directory as the
% folder containing the data (eg 'Fmr1 MWT Data')

%% Check for file conversion if necessary
% % TL can specify the while loop to run based on the electrode found in the
% % excel spreadsheet
% if ~isempty(varargin)
%     if ~strcmp(varargin{1}(end) , filesep)
%         varargin{1}(end+1) = filesep;
%     end
%     startF = 0;
%     while length(dir([varargin{1} , '\*.csv']) < 17
%         startF = startF + 1;
%     end
%     movefile([varargin{1} , '*'] ,pathname);
% else startF = 16;
% end
%     %Wait for file conversion to complete
%     
%     %Transfer tdt files to folder containing igor and whisking data
%     
%     % Run recording wrapper
%    if startF == 16; 
%% Arrange files and folders
% Embedded functions: none
%TL_arrangefiles(pathname); 

%% Convert recordings from csv to mat
% Embedded functions: none
%TL_tdtCSV2mat(pathname);

%% Convert event channels from ddt to mat
% Embedded functions: ddtChanRead
TL_ddt2mat(pathname , [1 2 3 4] , {'Sweep_Start' , 'Sweep_Info' , 'Licks' , 'Piezos'} , 1);

%%New conversion step
%%format change here

%% Form attributes file from tdt events channels and igor tables
% Embedded functions: TL_DecodeRawMatEvents , TL_electrode_channels , replacevals 
%temp = strfind(pathname , filesep); temp = temp(1);
%excel_path = [pathname(1:temp) 'data\kaeli\igor\Fmr1_MWT_Experiment_Info.xlsx']; clear temp;
%KJLTL_form_attributes_v6(pathname , excel_path , 1); 

% Filter and Common Average Reference spike recordings
% Embedded functions: none
%TL_CR_CAR_FILT_v2(pathname);
%%
%Arrange raw voltage files in the classic format
%TL_spike_sorter_v3(pathname, 3 , 'cref' , [1 2 3 4] , 1);
   end
