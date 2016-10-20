

clc
clear all



%% import data
cd .. 
cd data_bootstrap_input
load('EuropeanBanksData_bis.mat');
cd .. 
cd code

clearvars -except EuropeBank_data

%% Set parameters

rng(89);

%no. of nets for each ensemble
nNetworks = 1; 

%get number of time snapshots
t_steps = length(EuropeBank_data);


%% Cycle through all snapshots to find z and estimate ensembles

for t = 1:t_steps
    
    display(['Year ' num2str(EuropeBank_data(t).year)]);
    
    nNodes = length(EuropeBank_data(t).id);
    
    %% balance IB assets & liabilities
    [IB_in, IB_out] = f_balance_ib(EuropeBank_data(t).IBliabilities,...
        EuropeBank_data(t).IBassets);
    
   
    for n = 1:nNetworks
        
        %creat adj matrix
        adj = +(rand(nNodes, nNodes) <= 0.1);
        
        %assign weights
        [adj, actual_sum_rows, actual_sum_cols] = f_bootstrap_weights(adj, IB_in, IB_out);
        
        %export yearly data
        subfolder = num2str(EuropeBank_data(t).year);
        
        cd ..
        cd data_bootstrap_output
        if exist(subfolder, 'dir') ~= 7
            mkdir(subfolder);
        end
        cd(subfolder)
        
        flname = strcat(subfolder, 'exposure_net_ER', num2str(n), '.mat');
        save(flname, 'adj', 'actual_sum_rows', 'actual_sum_cols'); 
        
        cd ..
        cd ..
        cd code
        
        sentence = ['ensemble no.', num2str(n), ' of year ', num2str(EuropeBank_data(t).year),...
            ' exported in folder.'];
        disp (sentence);
        
    end
    
    
end
