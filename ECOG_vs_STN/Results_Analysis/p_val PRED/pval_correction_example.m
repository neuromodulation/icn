%addpath C:\code\wjn_toolbox
fname = 'p_val_rho_ECOGSTN.csv';
T=readtable(fname);

figure,
subplot(1,2,1)
plot(T.time_point_s_,T.rho_ECOG)
ylim([-1 0.5])
hold on
title('ECOG')
buc=sigbar(T.time_point_s_,T.p_ECOG<=0.05);
[p,i]= mypcluster(T.p_ECOG);
T.p_ECOG_clustercorrected = ones(size(T.p_ECOG));
T.p_ECOG_clustercorrected(i) = p;
bc=sigbar(T.time_point_s_,T.p_ECOG_clustercorrected<=0.05,'r');
plot(T.time_point_s_,T.rho_ECOG,'linewidth',1,'color','k');
legend([buc bc],{'P<0.05','P<0.05 corrected'});
subplot(1,2,2)
plot(T.time_point_s_,T.rho_STN)
ylim([-1 0.5])
hold on
title('STN')
buc=sigbar(T.time_point_s_,T.p_STN<=0.05);
[p,i]= mypcluster(T.p_STN);
T.p_STN_clustercorrected = ones(size(T.p_STN));
T.p_STN_clustercorrected(i) = p;
bc=sigbar(T.time_point_s_,T.p_STN_clustercorrected<=0.05,'r');
plot(T.time_point_s_,T.rho_STN,'linewidth',1,'color','k')
legend([buc bc],{'P<0.05','P<0.05 corrected'})
myprint('corrected')

writetable(T,['corrected_' fname])