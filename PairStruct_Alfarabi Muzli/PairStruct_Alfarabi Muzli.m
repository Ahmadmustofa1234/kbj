clc;

% File paths
img_path = 'Peppers.tiff';
pl_path = 'C:\Users\MSI GAMING\Downloads\Downloads\random-binary_70Kb.txt';

% Load Imageimg = imread(img_path);
img = imread(img_path);
[row, col] = size(img);
sz = numel(img)/2; % Total number of pixels

% Load Payload
SM = readmatrix(pl_path);
SM = reshape(SM, 1, []);  % Flatten the payload into a 1D array
SM = cast(SM, 'int32');    % Ensure it's in the correct data type

% Ensure the image has enough capacity to hold the payload
max_capacity = row * col;  % Maximum number of bits that can be embedded (1 bit per pixel)  % Potong payload sesuai kapasitas maksimum
payload_size = length(SM);


if payload_size > max_capacity
    error('Payload size exceeds image capacity!');
end

% Initialize variables
KT = zeros(row, col);  % Track bits embedded per pixel
stego = img;
idx2 = 1;  % Start indexing at 1
pi = zeros(2, sz);
pj = zeros(2, sz);
idx1 = 1;  % Initialize idx1 before the loop
for i = 1:row
   for j = 1:2:col % Assuming an even number of columns
       pi(1, idx1) = i; pi(2, idx1) = j;
       pj(1, idx1) = i; pj(2, idx1) = j + 1;
       idx1 = idx1 + 1;
   end
end

% Initialize difference array
D_arr = zeros(1, sz, 'int32');
for i = 1:sz
    D_arr(1, i) = int32(img(pj(1, i), pj(2, i))) - int32(img(pi(1, i), pi(2, i)));
end

% Define thresholds for adaptive embedding
base_threshold = 4;
max_threshold = 8;
low_threshold = 15;   % Standard deviation threshold for 1-bit embedding
medium_threshold = 20; % Threshold for 2-bit embedding
high_threshold = 30;  % Threshold for 3-bit embedding

% Adaptive embedding loop
for i = 1:row
    for j = 1:col
        if idx2 > length(SM)
            break;
        end

        % Define local neighborhood around the pixel
        local_patch = img(max(i-1,1):min(i+1,row), max(j-1,1):min(j+1,col));
        std_local = std2(local_patch);  % Calculate standard deviation

        % Determine the number of bits to embed based on local standard deviation
        if std_local <= low_threshold && abs(D_arr(1, i)) <= base_threshold
            bits_to_embed = 1;  % Embed 1 bit in smoother regions
        elseif std_local > low_threshold && std_local <= medium_threshold && abs(D_arr(1, i)) <= base_threshold
            bits_to_embed = 2;  % Embed 2 bits in moderate regions
        elseif std_local > medium_threshold && std_local <= high_threshold && abs(D_arr(1, i)) > base_threshold
            bits_to_embed = 2;  % Embed 3 bits in noisier regions
        else
            continue;
        end
        KT(i, j) = bits_to_embed;

        % Embed bits directly into the LSB(s) of the current pixel
        for b = 1:bits_to_embed
            if idx2 > length(SM)
                break;
            end
            stego(i, j) = bitset(stego(i, j), b, SM(idx2));
            idx2 = idx2 + 1;
        end
    end
end

% Display cover and stego images
% figure; imshow(img); title('Input: Cover image');
% figure; imshow(stego); title('Output: Stego image');

% Extraction
Secret_d = zeros(1, length(SM), 'int32');
idx3 = 1;  % Start from the first bit of the secret message
Stego_arr = zeros(1, sz, 'int32');

% Extraction loop (iterate over the pixels)
for i = 1:row
    for j = 1:col
        % Calculate the pair-pixel difference
        if i <= sz
            Stego_arr(1, i) = int32(stego(pj(1, i), pj(2, i))) - int32(stego(pi(1, i), pi(2, i)));
        end


        % Calculate local standard deviation around the pixel
        local_patch = img(max(i-1,1):min(i+1,row), max(j-1,1):min(j+1,col));
        std_local = std2(local_patch);  % Calculate standard deviation

        % Determine the number of bits to extract based on pair-pixel difference
        if std_local <= low_threshold && abs(Stego_arr(1, i)) <= base_threshold
            bits_to_extract = 1;  % Extract 1 bit from smoother regions
        elseif std_local > low_threshold && std_local <= medium_threshold && abs(Stego_arr(1, i)) <= base_threshold
            bits_to_extract = 2;  % Extract 2 bits from moderate regions
        elseif std_local > medium_threshold && std_local <= high_threshold && abs(Stego_arr(1, i)) > base_threshold
            bits_to_extract = 2;  % Extract 3 bits from noisier regions
        else
            continue;
        end

        % Extract the bits based on the calculated bits_to_extract
        for b = 1:bits_to_extract
            Secret_d(idx3) = bitget(stego(i, j), b);
            idx3 = idx3 + 1;
        end
    end
end

% Ensure the extracted message is the same length as the original payload
Secret_d = Secret_d(1:length(SM));
disp(length(Secret_d)); disp(length(SM));

% Check if the extracted message matches the original payload
if isequal(SM, Secret_d)
    disp('sama.');
else
    disp('ngga sama.');
end


% Calculate PSNR
thePSNR = psnr(stego, img); 
disp('PSNR Value');
disp(thePSNR);

% Embedding Capacity
EC = idx2 - 1; % Total embedded bits
disp('Embedding Capacity');
disp(EC);

% Payload Capacity
bpp = EC / (row * col);
disp('Payload Capacity (bits per pixel)');
disp(bpp);

% Extract the cover image (reset stego pixels back to original image)
extracted_cover = stego;
for i = 1:row
    for j = 1:col
        if KT(i, j) > 0
            extracted_cover(i, j) = img(i, j);
        end
    end
end

fprintf('SSIM dari Extracted Image dengan Cover Image: %f\n', ssim(img,stego));

num_errors = sum(Secret_d ~= SM);  % Count the number of incorrect bits
BER = num_errors / length(SM);  % Divide by total number of bits in the payload
disp('Bit Error Rate (BER):');
disp(BER);

% Calculate Recovery Rate (RR)
correct_recovered_bits = length(SM) - num_errors;  % Correctly recovered bits
RR = (correct_recovered_bits / length(SM)) * 100;  % Calculate the recovery rate in percentage
disp('Recovery Rate (RR) %:');
disp(RR);