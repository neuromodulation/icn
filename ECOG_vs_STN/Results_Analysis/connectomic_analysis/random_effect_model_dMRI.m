clear all, close all, clc

matlabbatch=[];
matlabbatch{1}.spm.stats.factorial_design.dir = {'C:\tmp\connectomics_ROIs\SPM_random_effect_dMRI'};
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac.name = 'dMRI';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac.dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac.variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac.ancova = 0;

%% Loop through hemispheres and fill in AvgR_Fz files from the C:\tmp\connectomics_ROIs\fMRI\ folder.
%files = dir('C:\tmp\connectomics_ROIs\fMRI\*')
files = dir('C:\tmp\connectomics_ROIs\dMRI\*')

%folder = 'C:\tmp\connectomics_ROIs\fMRI\';
folder = 'C:\tmp\connectomics_ROIs\dMRI\';

% sort files
ffiles = sort({files(:).name}');
files = sort_nat(ffiles);  % from https://de.mathworks.com/matlabcentral/fileexchange/10959-sort_nat-natural-order-sort
cnt = 0;
for sub = 0:14
    for sess = {'left', 'right'}
        file_cnt = 0;
        cellarr = [];
        for file = 1:size(files,1)
            % f = files{file};
            
            f = files(file);
            f = f{1};
            if size(f,2) < 3
                continue
            end
            %sub_f = str2num(f(5:7));  % for fMRI 
            sub_f = str2num(f(6:8)); % for DTI, since filenames start with ssub because of smoothing
            if contains(f,sess) == 1 && sub_f == sub
                file_cnt = file_cnt+1;
                cellarr{file_cnt} = strcat(folder, f);
            end
        end
        if size(cellarr,1) == 0
            continue
        else
            cnt = cnt + 1;
            matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.fsubject(cnt).scans = cellarr';  
            matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.fsubject(cnt).conds = ones(size(matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.fsubject(cnt).scans));
        end
    end
end
% cnt = 16 for 16 ECOG electrodes
matlabbatch{1}.spm.stats.factorial_design.des.fblock.maininters = {};
%%
matlabbatch{1}.spm.stats.factorial_design.cov.c = [0.658060030000000;0.711734480000000;0.549020863000000;0.623976546000000;0.571969597000000;0.597455084000000;0.292307677000000;0.352107469000000;0.243539604000000;0.148194472000000;0.362397572000000;0.339919038000000;0.507876205000000;0.454238142000000;0.413925157000000;0.388267266000000;0.335051810000000;0.208741421000000;0.115403022000000;0.0959871900000000;0.255036721000000;0.326226952000000;0.288791433000000;0.290079120000000;0.206994148000000;0.0894686560000000;0.316491719000000;0.251601613000000;0.459251391000000;0.484913139000000;0.419737826000000;0.203676762000000;0.121631851000000;0.0620716780000000;0.174597408000000;0.149999019000000;0.168008782000000;0.253791408000000;0.298088510000000;0.509607934000000;0.565925244000000;0.421615038000000;0.115658564000000;0.176490933000000;0.148283832000000;0.0687379130000000;0.0439634760000000;0.0468946120000000;0.256355138000000;0.174463015000000;0.256775235000000;0.376499491000000;0.291580852000000;0.644271360000000;0.559476473000000;0.544242209000000;0.193364220000000;0.182080122000000;0.110741585000000;0.0810761770000000;0.0369543100000000;0.0163621080000000;0.0134739400000000;0.00639995700000000;0;0;0;0;0.0192483810000000;0.0172430630000000;0.00193312300000000;0.0624229450000000;0.0449518660000000;0.0147927240000000;0.163216507000000;0.121120880000000;0;0;0.129685696000000;0.169117560000000;0.0867281370000000;0.0806320130000000;0.128188882000000;0.372052570000000;0.332659913000000;0.0548277950000000;0.0381693220000000;0.0451634570000000;0.00154861900000000;0.0638689180000000;0.0528243590000000;0.0682204840000000;0;0;0;0;0;0;0;0;0;0;0.153602940000000;0.115382072000000;0.159881873000000;0.201082252000000;0.198667600000000;0.260523988000000;0.279814823000000;0.220095890000000;0.278469914000000;0.280500538000000;0.263373437000000;0.265631629000000;0.216782673000000;0.121433496000000;0.106726870000000;0.102087614000000;0.117871276000000;0.172045155000000;0.261943801000000;0.280885020000000;0.338747069000000;0.361941607000000;0.198477576000000;0.273845248000000;0.274834513000000;0.313646416000000;0.243732983000000;0.204221832000000;0.124814705000000;0.0179061340000000;0.206642897000000;0.0274453970000000;0.138266719000000;0.192393553000000;0.148523327000000;0.0813199260000000;0.125288999000000;0.0811021900000000;0.136151297000000;0.141355936000000;0.121875115000000;0.0873273030000000;0.135320431000000;0.192362021000000;0.169448425000000;0.171610717000000;0.365936348000000;0.297459104000000;0.228720988000000;0.188668131000000;0.107862980000000;0.158905800000000;0.196378262000000;0.158720411000000;0.105572549000000;0.0725878460000000;0.000977902000000000;0.0402955950000000;0.0416498150000000;0.0120169690000000;0;0;0.196668834000000;0.247908528000000;0.442235016000000;0.190917049000000;0.00464753500000000;0;0.0615396530000000;0.00874960400000000;0.0258005060000000;0.0451998610000000;0.0176987810000000;0.00513983500000000;0.392282085000000;0.210456221000000;0;0.0659994900000000;0;0.0243758870000000;0.109875608000000;0.00533990800000000;0;0;0;0;0.111205709000000;0.0950733420000000;0.00669609500000000;0.0814116610000000;0.0803041890000000;0.0225516380000000;0.0503922370000000;0.0593775380000000;0.00907405700000000;0.0623578080000000;0.00565771500000000;0];
%%
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'Decoding performance';
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 2;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch);
