function [Events] = TL_ddt2mat(pathname , chans , evt_names , save_events)

%% FUNCTION: converts event channels from ddt to mat and saves. Detects if there are multiple ddt files and merges them if so

%% INPUTS: 
% [pathname] : pathname to recording. Should contain a folder 'evt' with
% events ddt file
% [chans] : channel numbers to convert
% [evt_names] : names of event channels in same order as chans

%% Set up variables
if ~strcmp(pathname(end) , filesep)
    pathname(end + 1) = filesep;
end

if ~iscell(evt_names)
    evt_names = {evt_names};
end

files=dir([pathname 'evt\' '*.ddt']);

filename = [pathname 'evt\' files.name]; clear temp;
%% Convert evts and save

for c = 1 : length(chans)
    [data , SamplingRate] = ddtChanRead(filename , chans(c) , [] , []);
    Events.(evt_names{c}) = data;
end
Events.Time = [1:length(data)] / SamplingRate;
Events.SampleRate = SamplingRate;

if save_events
save([pathname 'evt\' 'RawMatEvents.mat'] , 'Events','-v7.3');
end