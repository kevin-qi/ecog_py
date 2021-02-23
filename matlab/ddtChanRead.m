function [d,SamplingRate] = ddtChanRead(filename,Channel,ReadStart,ReadStop)
% ***CURRENTLY PRONE TO OUT OF MEMORY ERRORS
%ddtChanRead(filename,Channel) Read data from one channel of a .ddt file
%
% [d,SamplingRate] = ddtChanRead(filename,Channel,ReadStart,ReadStop)
%
% INPUT:
%   filename - if empty string, will use File Open dialog
%
% OUTPUT:
%   d - [npoints] data array

%if(nargin ~= 1)
%   disp('1 input argument is required')
%   return
%end


% nch = 0;
% npoints = 0;

if isempty(filename)
   [fname, pathname] = uigetfile('*.ddt', 'Select a ddt file');
	filename = strcat(pathname, fname);
end

fid = fopen(filename, 'r');
if(fid == -1)
    if ~exist(filename,'file')
       error('File not found: %s.',filename) 
    else
        error('Cannot open %s',filename);
    end
end

%disp(strcat('file = ', filename));
version = fread(fid, 1, 'int32');
dataoffset = fread(fid, 1, 'int32');
SamplingRate = fread(fid, 1, 'double');
nch = fread(fid, 1, 'int32');
year = fread(fid, 1, 'int32');
month = fread(fid, 1, 'int32');
day = fread(fid, 1, 'int32');
hour = fread(fid, 1, 'int32');
minute = fread(fid, 1, 'int32');
second = fread(fid, 1, 'int32');
gain = fread(fid, 1, 'int32');
comment = char(fread(fid, 128, 'char')');
BitsPerSample = fread(fid, 1, 'uchar');
ChannelGain = fread(fid, 64, 'uchar');
unused = fread(fid, 1, 'uchar');
MaxMagnitudeMV = fread(fid, 1, 'short');
RecordedChannelsList=find(ChannelGain~=255);
if Channel > nch
    error('Only %d channels detected and channel %d specified.',nch,Channel)
end


fseek(fid, 0, 1);

fsize = ftell(fid);
frewind(fid);
fseek(fid, dataoffset, 0);
fseek(fid, 2*(find(RecordedChannelsList==Channel)-1),0); %seek to desired channel

npoints = (fsize - dataoffset)/(nch*2);

if(nargin < 3) || (isempty(ReadStart) && isempty(ReadStop))
   ReadStart=1;
   ReadStop=npoints;
end
if(nargin < 4) || (~isempty(ReadStart) && isempty(ReadStop))
   ReadStop=npoints;
end

fseek(fid,(ReadStart-1)*2*nch,0); % seek to ReadStart
d = fread(fid, (ReadStop-ReadStart+1), 'int16=>int16',2*(nch-1)); %read in the channel wanted
%whos

fclose(fid);