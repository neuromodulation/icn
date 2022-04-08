'''
This file can convert the 6 microarray sample files from the Human Brain Mapping
using abagen https://abagen.readthedocs.io/en/stable/

This codes contains 3 trials with the built-in desikan killiany atlas
an atlas for which the csv was custom made Lausanne 33
and an atlas for which the csv and nii were made ourselves using matlab
Only the latter is documented compound_atlas_HCPex_SUIT_ABGT

JVH 05-04-2022
'''

import abagen

#################################### TEST CASES ######################################################################

from abagen import *
#ab = abagen.fetch_microarray(donors=['9861','10021'], data_dir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\genetics\HumanBrainMapping')

# atlas =  abagen.fetch_desikan_killiany()
#
# expression = abagen.get_expression_data(atlas['image'], atlas['info'], donors=['9861','10021'])
# print(expression)

# standard settings
# abagen.get_expression_data(atlas, atlas_info=None, *, ibf_threshold=0.5, probe_selection='diff_stability', donor_probes='aggregate', sim_threshold=None, lr_mirror=None, exact=None, missing=None, tolerance=2, sample_norm='srs', gene_norm='srs', norm_matched=True, norm_structures=False, region_agg='donors', agg_metric='mean', corrected_mni=True, reannotated=True, return_counts=False, return_donors=False, return_report=False, donors='all', data_dir=None, verbose=0, n_proc=1)
# https://abagen.readthedocs.io/en/stable/generated/abagen.get_expression_data.html#abagen.get_expression_data
#atlas =  abagen.fetch_desikan_killiany()
# atlas = dict()
# atlas['info'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Julians reorganization\Parcellations\compound_atlas_HCPex_SUIT_ABGT.csv'
# atlas['image'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Julians reorganization\Parcellations\compound_atlas_HCPex_SUIT_ABGT.nii.gz'
# expression = abagen.get_expression_data(atlas['image'], atlas['info'], donors=['9861','10021'], missing='interpolate', tolerance=0.5,  data_dir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\genetics\HumanBrainMapping\microarray')


# atlas = dict()
# atlas['info'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\LausanneAtlas\abagen_Lausanne33.csv'
# atlas['image'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\LausanneAtlas\region_masks.nii.gz'
# expression = abagen.get_expression_data(atlas['image'], atlas['info'], donors=['9861','10021'], tolerance=2, data_dir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\genetics\HumanBrainMapping\microarray')
# #
# print(expression)
# expression.to_csv('abagen_Lausanne33_expression_map.csv', encoding='utf-8')
#
# atlas= fetch_desikan_killiany()
# expression = abagen.get_expression_data(atlas['image'], atlas['info'], donors=['9861','10021'], missing='interpolate', tolerance=0.5,  data_dir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\genetics\HumanBrainMapping\microarray')
##########################################################################################################

# create the atlas instance
atlas = dict()
atlas['info'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Julians reorganization\Parcellations\compound_atlas_HCPex_SUIT_ABGT.csv'
atlas['image'] = r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Julians reorganization\Parcellations\compound_atlas_HCPex_SUIT_ABGT.nii.gz'

# set the expression following https://abagen.readthedocs.io/en/stable/generated/abagen.get_expression_data.html
# caution: atlas may not contain floats
expression = abagen.get_expression_data(atlas['image'], atlas['info'], donors='all', missing='interpolate', tolerance=2,  data_dir=r'C:\Users\Jonathan\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - PROJECT ALSP\Genetic_mapping_pipeline\genetics\HumanBrainMapping\microarray')

print(expression)

# save the gene expression map to csv
expression.to_csv('abagen_HCPex_SUIT_ABGT_expression_map.csv', encoding='utf-8')
