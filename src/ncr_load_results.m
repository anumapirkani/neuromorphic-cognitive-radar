function X = ncr_load_results(dirName)
%NCR_LOAD_RESULTS  Load saved experiment results (E1..E4) from a folder.
%   X = NCR_LOAD_RESULTS('results') returns a structure with the fields
%   E1, E2, E3, E4 for every result file present, ready for NCR_FIGURES and
%   NCR_TABLES.
if nargin < 1, dirName = 'results'; end
files = {'E1_main', 'E1'; 'E2_ablation', 'E2'; 'E3_interference', 'E3'; 'E4_latency', 'E4'};
X = struct();
for i = 1:size(files, 1)
    f = fullfile(dirName, [files{i, 1} '.mat']);
    if exist(f, 'file')
        L = load(f);
        X.(files{i, 2}) = L.S;
    end
end
end
