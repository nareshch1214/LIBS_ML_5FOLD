% Step 1: Load data from the Excel file
[fileName, filePath] = uigetfile('*.xlsx', 'Select the Excel file containing LIBS data');
fileFullPath = fullfile(filePath, fileName);

% Step 2: Read the data
data = readmatrix(fileFullPath);

% Step 3: Extract wavelength and intensity data
wavelength = data(:, 1);        % First column: Wavelength
intensities = data(:, 2:end);   % Remaining columns: Intensities

% Step 4: Group information
numSamples = 4;                % Number of samples: U0, U5, G0, G5
spectraPerSample = 15;         % Number of spectra per sample
sampleNames = {'U0', 'U5', 'G0', 'G5'};

% Step 5: Normalize the intensity data (optional, for PCA scaling)
intensitiesNorm = normalize(intensities, 'center', 'mean', 'scale', 'std'); % Z-score normalization

% Step 6: Apply PCA
[coeff, scores, explained] = pca(intensitiesNorm');

% Step 7: Visualize PCA Results (3D Scores Plot with Different Symbols)
figure;
hold on;
colors = lines(numSamples); % Different colors for each sample
markers = {'o', '^', '*', 's'}; % Circle, triangle, star, square

% Plot scores for the first three principal components
for i = 1:numSamples
    idxStart = (i-1)*spectraPerSample + 1;
    idxEnd = i*spectraPerSample;
    scatter3(scores(idxStart:idxEnd, 1), scores(idxStart:idxEnd, 2), scores(idxStart:idxEnd, 3), ...
        100, colors(i, :), markers{i}, 'filled', 'DisplayName', sampleNames{i});
end

% Customize the 3D plot
xlabel(['PC1 (' num2str(explained(1), '%.2f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.2f') '%)']);
zlabel(['PC3 (' num2str(explained(3), '%.2f') '%)']);
title('3D PCA of LIBS Spectra');
legend('Location', 'best');
grid on;
view(45, 30); % Adjust view angle
hold off;

% Step 8: Bar plot of explained variance
figure;
bar(explained(1:10), 'FaceColor', [0.2 0.6 0.8]);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Explained Variance by Principal Components');
