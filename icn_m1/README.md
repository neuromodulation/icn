The icn_m1 module allows for real time processing of multimodal neurophysiological data.
It runs in an anaconda environment. The necessary path dependencies can be install using the /env/environment.yaml file:

```
conda env create -f environment.yml
```

The primary use of the M1 toolbox is to predict movement from electrophysiological data. It is intended to be used for adaptive Deep Brain Stimulation (see https://pubmed.ncbi.nlm.nih.gov/30607748/ for the general concept).

The toolbox has been tested with STN-LFP and ECOG data from Parkinson's disease patients, and prediction of movements without individual training had been shown above chance level.


The concept is based on the iEEG BIDS format: https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/04-intracranial-electroencephalography.html. This standard is extended by "M1_channel.tsv" files which are added per individual run. By this each channel is assigned with different variables:

 - name: the channel name (taken from "run" channels.tsv files)
 - rereference: e.g. average, bipolar, variable
 - used: 1/0 Flag indicating if this channel should be used in the pipeline
 - target: 1/0 Flag indicating a predictor/data stream variable

This exemplary M1_channels.tsv file shows how the setup is implemented for a run file with ECOG and STN recordings:

|name|rereference|used|target|
|:---|:---|:---|:---|
|STN_LEFT_0|average|1|0|
|STN_LEFT_1|average|1|0|
|STN_LEFT_2|average|1|0|
|ECOG_LEFT_0|average|1|0|
|ECOG_LEFT_1|average|1|0|
|ECOG_LEFT_2|average|1|0|
|ECOG_LEFT_3|average|1|0|
|ECOG_LEFT_4|average|1|0|
|ECOG_LEFT_5|average|1|0|
|ECOG_LEFT_6|average|1|0|
|ECOG_LEFT_7|average|1|0|
|MOV_RIGHT|average|1|1|
|MOV_LEFT|average|1|1|
|MOV_RIGHT_CLEAN|average|1|1|
|MOV_LEFT_CLEAN|average|1|1|

Based on these files all channels are selected which will be used in the ongoing analysis.
In the *settings* folder different parameters are specified for the pipeline to project patient individual coordinates to a common grid, which is also specified for the cortex and subcortex in the settings folder.

The pipeline specific parameters are defined in settings/settings.json as follows:
```json
{
    "BIDS_path": "BIDS_PATH_LOCATION",
    "out_path": "PIPELINE_OUTPUT_PATH_LOCATION",
    "resamplingrate": 10,
    "max_dist_cortex": 20,
    "max_dist_subcortex": 5,
    "normalization_time": 10,
    "frequencyranges": [[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]],
    "seglengths": [1, 2, 2, 3, 3, 3, 10, 10, 10]
}
```

Here the parameter *resamplingrate* [s] defines the resulting sampling frequency.
The parameters *max_dist_cortex* [mm] and *max_dist_subcortex* [mm] specify the interpolation distance of patient-individual channels into common grid points.
The *normalization_time* is used in order to define in which time the data stream is being normalized in real time (by default implemented by the *median*).
*frequencyranges* [Hz] defines designated frequency bands which will be extracted through band-filtering. *seglengths* [Hz] defines which time window is being used with respect to the upper defined *frequencyranges*.

In the upper example the sampling frequency is specified as 10 Hz, eight frequency bands are defined. For alpha band power is extracted in a range of 1 s, while for the highest specified frequency band (high gamma) 100 ms is used.  


When the settinngs are defined, the pipeline_runall.py can simply be launched in the upper used environment:

```
python pipeline_runall.py
```

This will run through the defined BIDS directory and write out *pickle* files including a *dictionary* containing the following keys in the settings-defined *out_path*:

|Key      |type                          |explanation                                                                                               |
|---------|------------------------------|----------------------------------------------------------------------------------------------------------|
|vhdr_file| string                       | name of pojected *.vhdr* file                                                                            |
|resamplingrate| int                          | defined resamping rate in settings                                                                       |
|projection_grid| numpy array                  |projection grid of cortex and subcortex in list Cortex LEFT, Subcortex LEFT, Cortex RIGHT, Subcortex RIGHT|
|subject  | string                       | BIDS subject ID                                                                                          |
|run      | string                       | BIDS run                                                                                                 |
|sess     | string                       | BIDS session                                                                                             |
|sess_right| boolean                      | definition if session acquisition had been from left or right hemisphere                                 |
|used_channels| list                         | used channels from *M1_channels.tsv"*                                                                    |
|coord_patient| numpy array                  | patient individual coordinates from                                                                      |
|proj_matrix_run| numpy array                  | projection matrix for cortex and subcortex from patient individual to common grid points                 |
|fs       | int                          | sampling frequency of recorded channels                                                                  |
|line_noise| int                          | line noise                                                                                               |
|seglengths| list                         | settings specified acquisition time for band power calculation                                           |
|normalization_samples| int                          | used samples for normalization                                                                           |
|new_num_data_points| int                          | number of data points for projected data                                                                 |
|downsample_idx| numpy array                  | indices of samples used for real time feature extraction                                                 |
|filter_fun| numpy array                  | calculated filter banks                                                                                  |
|offset_start| int                          | number of data points skipped for uniform number of features in first sample                             |
|arr_act_grid_points| numpy array                  | boolean array indicating active / projected grid points of cortical and subcortical common grid          |
|rf_data_median| numpy array                  |non-projected, band-filtered, resampled datastream                                                        |
|**pf_data_median**| numpy array                  |projected, band-filtered, resampled datastream                                                            |
|**label_baseline_corrected**| numpy array                  |normalized, baseline corrected label array                                                                |
|label    | numpy array                  | original labels                                                                                          |
|label_con_true| numpy array                  | definition of contralateral labels                                                                       |

Based on the projected data a common *electrode* space exists across subjects. This allows for predictions across subjects using bandpass-filtered features. When the BIDS dataset consists of multiple subjects a *leave-1-patient-out* Cross Validation can be performed. Here the test dataset is a left out patient.

Since every patient is assumed to have a different number of active grid points, the Cross Validation needs to be performed for every single grid point individually. A conventional cross validation thus might have a lot of read and write operations. Therefore two different types of Cross Validation pipelines are defined:
 - leave_one_out_CV.py
 - CV_load_all_RAM.py

 In *CV_load_all_RAM.py* all interpolated runs are loaded in the RAM, and then the Cross Validation is performed based on the loaded dictionaries. For the *leave_one_out_CV.py* all files are individually loaded and the respective train and test sets are constructed sequentially.
