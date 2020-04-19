The icn_m1 module allows for real time processing of multimodal neurophysiological data. 
It runs in an anaconda environment. The neccessary path dependencies can be install using the /env/environment.yaml file: 

```
conda env create -f environment.yml
```

The module is based on the BIDS format: https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/04-intracranial-electroencephalography.html. 

In this preliminary version "left" and "right" sessions are read respective from the "left" and "right" hemisphere recordings. An already trained model can be loaded to predict the provided dataset.
In the icn_m1/settings/settings.json the BIDS folder is specified, as well as real time resampling parameters (new sampling rate, frequency bands, normalization time, interpolation parameters).

The icn_m1 toolbox reads respective channel names and interpolates them to a corresponding grid, provided by the icn_m1/settings/ .tsv files. 