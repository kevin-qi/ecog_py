import numpy as np
import pandas as pd

class Experiment:
    """
    Data structure for handling data from an ECoG experiment
    
    Attributes
    ----------
    exp_name : string
        Experiment Name
    segments : list<Segment>
        List of Segment objects in the Experiment. 
    headers_df : pandas.DataFrame
        DataFrame containing header.csv information (Loaded from file using DataLoader module)
    trials_df : pandas.DataFrame
        DataFrame containing trials.csv information (Loaded from file using DataLoader module)    
    stimuli_df : pandas.DataFrame
        DataFrame containing stimuli.csv information (Loaded from file using DataLoader module)
    num_trials : int
        Number of trials
    has_laser : bool
        True if using optogenetics (trials.csv contains has_laser field)
    """
    def __init__(self, headers_df, trials_df, stimuli_df):
        """
        Initialize Experiment object from pandas DataFrame containing header.csv information
        
        Parameters
        ----------
        headers_df : pandas.DataFrame
            DataFrame containing header.csv information (Loaded from file using DataLoader module)
        trials_df : pandas.DataFrame
            DataFrame containing trials.csv information (Loaded from file using DataLoader module)    
        stimuli_df : pandas.DataFrame
            DataFrame containing stimuli.csv information (Loaded from file using DataLoader module)
        has_laser : bool
            True if using optogenetics (trials.csv contains has_laser field)
        """

        self.exp_name = headers_df.iloc[0]['ExptName']
        
        # Check DataFrames are valid
        self._is_valid_header_df(headers_df)
        self._is_valid_stimuli_df(stimuli_df)
        self._is_valid_trials_df(trials_df)
        
        # Populate Adrian information
        self.headers_df = headers_df
        self.trials_df = trials_df
        self.stimuli_df = stimuli_df

        # Populate Experiment Segments
        self.segments = self._load_segments_info(self.headers_df, self.trials_df)

        # Populate some useful variables
        self.num_trials = np.max(trials_df['TrNum'])

        

    def __repr__(self):
        return 'Experiment: {}\nNumber of segments: {}\nOptogenetics: {}\nFields:[{}]'.format(
            self.exp_name,
            len(self.segments),
            self.has_laser,
            'exp_name, headers_df, trials_df, stimuli_df, segments, num_trials, has_laser'
        )     
        
    def _load_segments_info(self, headers_df, trials_df):
        """
        Load header info from header_df
        
        """
        sweep_period = int(np.around(np.mean(np.diff(trials_df['TrStartTime']))/1000, decimals=1)*1000)
        sweep_duration = int(np.around(np.mean(np.array(trials_df['TrEndTime']) - np.array(trials_df['TrStartTime']))/1000, decimals=1)*1000)

        segments = []
        for i, row in headers_df.iterrows():
            segment_num = int(row['SegmentNum'])
            repetitions = None
            stim_set = None
            stim_layout = int(row['StimLayout'])
            first_trial_num = int(row['FirstTrialNum'])
            last_trial_num = int(row['LastTrialNum'])
            

            whisker_map = np.zeros(int(row['Nelements'])).astype(str)
            stim_duration = np.zeros(int(row['Nelements'])).astype(int)
            stim_ampl = np.zeros(int(row['Nelements'])).astype(int)
            stim_shape = np.zeros(int(row['Nelements'])).astype(int)
            for piezo_num in range(int(row['Nelements'])):
                if(not int(row['ElemPiezo{}'.format(piezo_num)]) == piezo_num):
                    print("WARNING: ElemPiezo{} does not match {}. Whisker map may be incorrect - double check this!".format(piezo_num, piezo_num))
                
                # Whisker map
                whisker_map[piezo_num] = row['PiezoLabel{}'.format(piezo_num)]
                
                if(int(row['ElemShape{}'.format(piezo_num)]) != 2):
                    print("WARNING: ElemShape = {} is not tested, stimuli duration might be incorrect!".format(piezo_num))
                
                # Stimulus Shape
                stim_shape[piezo_num] = int(row['ElemShape{}'.format(piezo_num)])
                
                # Stimulis Duration
                rise = int(row['ElemRise{}'.format(piezo_num)])
                hold = int(row['ElemDur{}'.format(piezo_num)])
                stim_duration[piezo_num] = rise + hold + rise
                
                # Stimulis Amplitude
                stim_ampl[piezo_num] = int(row['ElemAmp{}'.format(piezo_num)])
                
                
            # Stimuli info is slightly different for different StimLayouts
            if stim_layout == 5: # RF Mapping
                stim_train_N = int(row["RFStimN"])
                stim_isi = int(row['RFStimISI'])
                stim_interval = int(row['RFStimInterval'])
                stim_onset = int(row['RFStimOnset'])
                self.has_laser = None
            elif stim_layout == 1: # Standard
                stim_train_N = int(row["StdStimN"])
                stim_isi = int(row['StdStimISI'])
                stim_interval = int(row['StdStimISI']) # stim_interval is stim_isi for standard layout (1 stim train per sweep)
                stim_onset = int(row['StdStimOnset'])
                self.has_laser = None
            elif stim_layout == 7: # Standard
                stim_train_N = int(row['MWStimN'])
                stim_isi = int(row['MWStimISI'])
                stim_interval = int(row['MWStimBurstISI']) # stim_interval is stim_isi for standard layout (1 stim train per sweep)
                stim_onset = int(row['MWStimOnset'])
                self.has_laser = True
            else:
                stim_train_N = int(row["StdStimN"])
                stim_isi = int(row['StdStimISI'])
                stim_interval = int(row['StdStimISI']) # stim_interval is stim_isi for standard layout (1 stim train per sweep)
                stim_onset = int(row['StdStimOnset'])
                self.has_laser = None
                print("WARNING: stim_layout {} may not be supported".format(stim_layout))
                
                
            segments.append(Segment(
                 exp_name = self.exp_name, 
                 segment_num=segment_num, 
                 repetitions=repetitions, 
                 stim_set=stim_set, 
                 stim_layout=stim_layout, 
                 sweep_duration=sweep_duration, 
                 sweep_period=sweep_period,
                 first_trial_num=first_trial_num,
                 last_trial_num=last_trial_num,
                 stim_interval=stim_interval,
                 stim_onset=stim_onset,
                 stim_isi=stim_isi,
                 stim_train_N=stim_train_N,
                 stim_shape=stim_shape, 
                 stim_ampl=stim_ampl,
                 stim_duration=stim_duration,
                 whisker_map=whisker_map,
                 trials_df=self.trials_df,
                 stimuli_df=self.stimuli_df,
                 has_laser=self.has_laser
            ))
            
        return segments
    
    def _is_valid_header_df(self, df):
        for col in ['ElemShape0', 'ElemDur0', 'ElemPiezo0', 'PiezoLabel0', 'FirstTrialNum', 'LastTrialNum', 'SegmentNum', 'StimLayout']:
            if col not in df:
                raise TypeError('Header pd.DataFrame is not valid. Missing {} column'.format(col))
    
    def _is_valid_stimuli_df(self, df):
        for col in ['Trial', 'StimElem', 'Time_ms', 'Ampl']:
            if col not in df:
                raise TypeError('Stimuli pd.DataFrame is not valid. Missing {} column'.format(col))
    
    def _is_valid_trials_df(self, df):
        for col in ['TrNum', 'Segment', 'TrStartTime', 'TrEndTime', 'StimNum', 'StimOnsetTime']:
            if col not in df:
                raise TypeError('Trials pd.DataFrame is not valid. Missing {} column'.format(col))
    
class Segment:
    """
    Data structure for storing trial information and data
    
    Attributes
    ----------
    exp_name : string
        Experiment Name
    segment_num : int
        Segment number
    repetitions : int
        Number of repetitions (A repetition is N sweeps, in which the full stimulus set is presented once).
    stim_set : ndarray of type int
        Array of stimulus IDs in stimulus set
    stim_layout : int
        Stimulus layout ID (1 - Standard, 2 - Trains, 3 - Interleaved, 5 - RF Mapping, 7 - Multi-whisker synchronous stimuli)
    sweep_duration : int (milliseconds)
        Length of trials in ms (Must be shorter than sweep_period)
    sweep_period : int (milliseconds)
        Inter-sweep-interval. Interval from start of trial N to start of trial N + 1 in ms. Must be > 200ms longer than sweep_duration.
    first_trial_num : int 
        Index of first trial in this segment
    last_trial_num : int
        Index of last trial in this segment
    stim_interval : int (milliseconds)
        Interval between stimulus units
    stim_onset : int (milliseconds)
        Stimuli onset time
    stim_train_N : int 
        Number of stimuli per train
    stim_isi : int (milliseconds)
        Stimuli train interval
    stim_shape : list<int>
        Stimulus waveform shape (2 - Ramp, 3 - Cosine-rise, 4 - Shulz CF1, 5 - Vibrotactile (independent), 6 - Vibrotactile (identical))
    stim_ampl : list<int>
        Piezo stimulus amplitude in (micrometers)
    stim_duration: list<int>
        Piezo stimulus duration (rise + hold + rise)
    trials : list<Trial>
        List of trials in this segment
    """
    
    def __init__(self,
                 exp_name, 
                 segment_num, 
                 repetitions, 
                 stim_set, 
                 stim_layout, 
                 sweep_duration, 
                 sweep_period,
                 first_trial_num,
                 last_trial_num,
                 stim_interval,
                 stim_onset,
                 stim_isi,
                 stim_train_N,
                 stim_shape,
                 stim_ampl,
                 stim_duration,
                 whisker_map,
                 trials_df,
                 stimuli_df,
                 has_laser):
        self.exp_name = exp_name
        self.segment_num = segment_num
        self.reptitions = repetitions
        self.stim_set = stim_set
        self.stim_layout = stim_layout
        self.sweep_duration = sweep_duration
        self.sweep_period = sweep_period
        self.first_trial_num = first_trial_num
        self.last_trial_num = last_trial_num
        self.stim_shape = stim_shape
        self.stim_ampl = stim_ampl
        self.stim_duration = stim_duration
        self.whisker_map = whisker_map
        self.stim_onset = stim_onset
        self.stim_isi = stim_isi
        self.stim_train_N = stim_train_N
        self.stim_interval = stim_interval
        self.stimuli_df = stimuli_df
        self.has_laser = has_laser
        self.trials = self._load_trial_info(trials_df)
        
        
    def __repr__(self):
        segment_repr = 'segment_num: {}\n'.format(self.segment_num)
        trial_repr = 'sweep_duration: {}\nsweep_period: {}\nfirst_trial_num: {}\nlast_trial_num: {}\n'.format(
            self.sweep_duration,
            self.sweep_period,
            self.first_trial_num,
            self.last_trial_num
        )
        stim_repr = 'stim_layout: {}\nstim_onset: {}\nstim_isi: {}\nstim_interval: {}\nstim_train_N: {}\nstim_duration: {}\nwhisker_map: {}\n'.format(
            self.stim_layout,
            self.stim_onset,
            self.stim_isi,
            self.stim_interval,
            self.stim_train_N,
            self.stim_duration,
            self.whisker_map
        )
        return segment_repr + '\n' + trial_repr + '\n' + stim_repr
    
    def _load_trial_info(self, trials_df):
        """
        Load trial info from trials_df
        """
        trials = []
        for i, row in trials_df.iterrows():
            if(int(row['Segment']) == self.segment_num): # If trial is part of this segment
                trial_num = int(row['TrNum'])
                
                # Ensure stimlayout matches between header.csv and trials.csv
                # assert int(row['StimLayout']) == self.stim_layout, "StimLayout does not match between header.csv (stimlayout {}) and trials.csv (stimlayout {})".format(self.stim_layout, int(row['StimLayout']))
                
                # Ensure trial num is in range of first_trial_num and last_trial_num from header.csv
                assert int(row['TrNum']) >= self.first_trial_num and int(row['TrNum']) <= self.last_trial_num, "Unexpected Trial Num, out of range [first_trial_num, last_trial_num]"
                
                trials.append(Trial(
                    exp_name = self.exp_name,
                    segment_num = self.segment_num,
                    trial_num = trial_num,
                    sweep_duration = self.sweep_duration,
                    sweep_period = self.sweep_period,
                    stim_layout = self.stim_layout,
                    stim_onset = self.stim_onset,
                    stim_isi = self.stim_isi,
                    stim_train_N = self.stim_train_N,
                    stim_shape = self.stim_shape,
                    stim_interval = self.stim_interval,
                    stim_duration = self.stim_duration,
                    stim_ampl = self.stim_ampl,
                    whisker_map = self.whisker_map,
                    stimuli_df = self.stimuli_df,
                    laser = int(row['Laser'])
                ))
        return trials
    
class Trial:
    """
    Data structure for storing sweep information and data
    """
    def __init__(self, 
                 exp_name, 
                 segment_num, 
                 trial_num, 
                 sweep_duration, 
                 sweep_period, 
                 stim_layout,
                 stim_interval,
                 stim_onset,
                 stim_isi,
                 stim_train_N,
                 stim_shape, 
                 stim_ampl,
                 stim_duration,
                 whisker_map,
                 stimuli_df,
                 laser):
        self.exp_name = exp_name
        self.segment_num = segment_num
        self.trial_num = trial_num
        self.sweep_duration = sweep_duration
        self.sweep_period = sweep_period
        self.stim_layout = stim_layout
        self.stim_shape = stim_shape
        self.stim_ampl = stim_ampl
        self.stim_onset = stim_onset
        self.stim_isi = stim_isi
        self.stim_train_N = stim_train_N
        self.stim_interval = stim_interval,
        self.stim_duration = stim_duration,
        self.stim_ampl = stim_ampl,
        self.whisker_map = whisker_map
        self.laser = laser
        self.stim_elems, self.stim_whiskers = self._load_stimuli_info(stimuli_df)
            

    def _load_stimuli_info(self, stimuli_df):
        """
        Load stimuli info from stimuli_df
        """
        stimuli = []
        whisker = []
        for _, row in stimuli_df[stimuli_df['Trial'] == self.trial_num].iterrows():
            stim_elem = int(row['StimElem'])
            
            stimuli.append(stim_elem)
            whisker.append(self.whisker_map[stim_elem])


        return stimuli, whisker

    def __repr__(self):
        trial_repr = 'trial_num: {}\nsweep_duration: {}\nsweep_period: {}\n'.format(
            self.trial_num,
            self.sweep_duration,
            self.sweep_period,
        )
        stim_repr = 'stim_layout: {}\nstim_onset: {}\nstim_isi: {}\nstim_interval: {}\nstim_train_N: {}\nstim_duration: {}\nwhisker_mp: {}\nstim_elems: {}\nstim_whiskers: {}'.format(
            self.stim_layout,
            self.stim_onset,
            self.stim_isi,
            self.stim_interval,
            self.stim_train_N,
            self.stim_duration,
            self.whisker_map,
            self.stim_elems,
            self.stim_whiskers
        )
        return trial_repr + '\n' + stim_repr