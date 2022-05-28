clear all
close all

dir_to  = '/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/test_1/2_SLICES/mat';
T0      = 20;       % Cº
delta0  = 0;        % mm
t       = 2e-3;     % m
L       = 300e-3;   % m
alpha   = 24;       % µm/(m·K)


% Look subfolders
files = dir(fullfile(dir_to, '*.mat'));
[files_val, files_idx] = natsortfiles({files.name});

% Sort by number
for i=1:length(files)
    new_files(i) = files(files_idx(i));
end
files = new_files;

% Load into a struct
for i=1:length(files)
    a(i) = load([files(i).folder '/' files(i).name]);
end

% Add delta_T and delta_EPS column according to ref_file
for i=1:length(a)
    delta_T         = a(i).temperature-T0             ;% K
    delta_flecha    = (a(i).flecha-delta0) * 1e-3     ;% mm to m
    x               = a(i).x * 1e-3                   ;% mm to m

    eps_mec = 3*delta_flecha*t/(2*L^3) * (L-x) * 1e6                ;% Mechanical microdeformations
    eps_the = alpha * delta_T                                       ;% Thermal  microdeformations
    delta_EPS = eps_mec + eps_the                                   ;% Total microdeformations

    a(i).delta_T    = delta_T;
    a(i).delta_EPS  = delta_EPS;
end

save('slices.mat', 'a');