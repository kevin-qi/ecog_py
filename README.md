
[![Documentation Status](https://readthedocs.org/projects/ecog-py/badge/?version=latest)](https://ecog-py.readthedocs.io/en/latest/?badge=latest)

# ecog_py
**ecog_py** is a ECoG processing pipeline that builds on top of Bouchard Lab's [process_nwb](https://github.com/BouchardLab/process_nwb) ECoG pre-processing library. The primary purpose of this codebase is to analyze uECoG signals from mice whisker S1. 

# Dependencies

- [conda](https://docs.conda.io/en/latest/)
- [process-nwb](https://github.com/BouchardLab/process_nwb)
	- ECoG processing library by the Bouchard lab. This must be installed from source as the pip version is not always up to date.

# Installation
```bash
# Clone ecog_py repo
$ git clone https://github.com/kevin-qi/ecog_py.git
$ cd ecog_py/

# Build conda env from environment.yml
$ conda env create -f environment.yml

# Clone process_nwb repo (make sure you are still inside ecog_py/)
$ git clone https://github.com/BouchardLab/process_nwb.git
$ cd process_nwb

# Build process_nwb environment 
$ conda env update --file environment.yml
$ pip install -e .
```

Your final project directory should look something like this
```bash
ecog_py/ # project root
  ...
  ecog_py/
  matlab/
  process_nwb/
```

# System Requirements
This pipeline has some pretty hefty RAM requirements because wavelet transformation splits every ECoG channel into 54 bands. It is recommended to have at least 32gb of RAM or more for a smooth experience, but you could probably get by with 16gb if you try hard enough and your dataset is small (less than 1gb). The key parameter for tuning RAM usage is the final downsampling factor **after** wavelet transformation, `post_ds_factor`.  The recommended `post_ds_factor` is roughly $$\text{post\_ds\_factor} \approx round(\frac{2 * 54 *\text{raw\_ecog\_RAM} * \frac{\text{fs}}{\text{fsds}}}{\text{available\_RAM}})$$
where `fs` is the original sampling frequency of your ECoG data (probably ~24kHz or 48kHz), `fsds` is the downsampled frequency (defaults to 3000hz), and RAM is measured in gigabytes. If your computer freezes, try increasing the `post_ds_factor`. 

# Pipeline Usage

This ECoG processing pipeline requires a standardized data format. You may need to write some code (in any language) to convert your data into an accepted format.

## Pipeline Input Format

- **ECoG Data**
	- `*TDT/_ch{}.csv`
		- TDT output files. Should be 1 csv file per channel.
- **Trial Timing**
	- `evt/*_Evnt.ddt`
		- Event .ddt file from TDT

- **Trial Info**
	- `Adrian/*.csv`
		- This is the `Trials.csv` file output from Igor
			- Should look something like this:
				```
				TrNum	Segment	ISS0Time	Arm0Time	TrStartTime	TrEndTime	RWStartTime	RWEndTime	StimOnsetTime	StimNum	Tone	TrType	LickInWindow	TrOutcome	RewardTime	NLicks	CumNRewards	CumVol	StimLayout	StimOrder	Unused
				0	1	159085	-90	-84	11948	9942	11947	-84	5	0	0	0	2	0	0	0	0	5	0	0
				1	1	159085	13250	13255	25286	23280	25285	13255	6	0	0	0	2	0	0	0	0	5	0	0
				2	1	159085	26589	26595	38626	36620	38625	26595	7	0	0	0	2	0	0	0	0	5	0	0
				3	1	159085	39929	39934	51965	49959	51964	39934	4	0	0	0	2	0	0	0	0	5	0	0
				4	1	159085	53198	53203	65234	63228	65233	53203	1	0	0	0	2	0	0	0	0	5	0	0
- **Stimulis Info**
	- `Adrian/*.csv`
		- This is the `Stimuli.csv` file output from Igor
			- Should look something like this:
				```
				Trial Posn StimElem Time_ms Ampl
					0	0	0	0	180.00
					0	1	1	100	180.00
					0	2	2	200	180.00
					0	3	3	300	180.00
					0	4	4	400	180.00
					0	5	5	500	180.00
					0	6	6	600	180.00
					0	7	7	700	180.00
					1	0	0	0	180.00
- **Header Info**
	- `Adrian/*.csv`
		- This is the `Header.csv` file output from Igor
			- Should look something like this:
				```
				ExptName	TestFile
				ExptCode	Default
				SegmentNum	1
				ISS0time	159085
				FirstTrialNum	0
				LastTrialNum	34
				StimLayout	5
				Nstimuli	1
				StimOrder	0
				ArduinoMode	4
				ITImean	1.00
				ITIrange	0.00
				MaxLicksLimit	100
## Project Directory Structure
Once you have all the data in the accepted formats, place them in your project directory as follows.
```
data/
  <your_exp_name>/
    TDT/
      <your_ecog_data>.<csv> # Should be 32 of these for 32 channel recordings
    Adrian/
      <your_trial_info>.csv
      <your_stimulis_info>.csv
      <your_header_info>.csv
    evt/
      <your_event_info>.ddt # Event info from TDT in .ddt format
  
ecog_py/
matlab/
```
## Pipeline Usage
Now you are ready to get started! The pipeline template is provided in `ecog_py_pipeline_template.ipynb`. Anytime you want to run the pipeline on a new dataset, simply make a copy of the template and follow all the steps in the pipeline, filling in any required variables and parameters. The pipeline should walk you through all the steps in the analysis.

For more detailed documentation, check out the [ecog_py docs](https://ecog-py.readthedocs.io/en/latest/) and [process_nwb docs](https://process-nwb.readthedocs.io/en/latest/).
