addpath('C:\\Users\\ICN_admin\\Dropbox (Brain Modulation Lab)\\wjn_toolbox');

T = readtable("df_all.csv");
T_ECOG = T(strcmp(T.loc, "ECOG"),:);
T_STN = T(strcmp(T.loc, "STN"),:);

% Fisher Transform performances 
T_ECOG.r2_conZ = fisherZ(T_ECOG.r2_con);
T_ECOG.r2_ipsZ = fisherZ(T_ECOG.r2_ips);
fitlme(T_ECOG, 'r2_conZ ~ dist_con + (1|sub)')
fitlme(T_ECOG, 'r2_ipsZ ~ dist_ips + (1|sub)')

fitlme(T_ECOG, 'r2_conZ ~ b_peak + (1|sub)')
fitlme(T_ECOG, 'r2_ipsZ ~ b_peak + (1|sub)')

fitlme(T_ECOG, 'r2_conZ ~ UPDRS_rigidity_upper_extrimity_contralateral + (1|sub)')
fitlme(T_ECOG, 'r2_ipsZ ~ UPDRS_rigidity_upper_extrimity_ipsilateral + (1|sub)')

fitlme(T_ECOG, 'r2_conZ ~ UPDRS_combined_akinesia_rigidty_contalateral + (1|sub)')
fitlme(T_ECOG, 'r2_ipsZ ~ UPDRS_combined_akinesia_rigidty_ipsilateral + (1|sub)')

fitlme(T_ECOG, 'r2_conZ ~ UPDRS_total + (1|sub)')
fitlme(T_ECOG, 'r2_ipsZ ~ UPDRS_total + (1|sub)')


% fit LM best ch 
T = readtable("df_ECOG_CON.csv");
T.r2_conZ = fisherZ(T.r2_con);
T.r2_ipsZ = fisherZ(T.r2_ips);
fitlm(T, 'r2_conZ ~ dist_con')


