function [volumes, segmented_img] = segment_brain_csf(in_file, mask_file, ex_segmentation_file)
%SEGMENT_BRAIN_CSF Segment T2w rat brain MRI into 'brain' and 'csf'
% 
%   [volumes, segmented_img] = SEGMENT_BRAIN_CSF(IN_FILE, MASK_FILE, 
%   EX_SEGMENTATION_FILE) segments NIFTI input image IN_FILE into 4 labels
%   corresponding those given in EX_SEGMENTATION_FILE. 
%
%   IN_FILE must be motion corrected, N4 bias corrected, and binned so that
%   nii(:,:,:,1) is pre-HTS image and nii(:,:,:,2) is post-HTS. 
%
%   MASK_FILE is a brain mask
% 
%   EX_SEGMENTATION_FILE is an example segmentation of IN_FILE with labels 
%       1: Blood vessel
%       2: Gray matter
%       3: White matter
%       4: CSF
%
%   
%   Copyright 2022 Kristian Nygaard Mortensen, kristiannm@drcmr.dk

% Read T2w MRI. 
nii = niftiread(in_file);
hdr = niftiinfo(in_file);

% Read brain mask, erode by 1 voxel to avoid surface  partial volume
% effects.
mask = imerode(niftiread(mask_file), strel('sphere', 1));

% Read example segmentation of tissues.
seg = niftiread(ex_segmentation_file);

for i = 1:2
    
    % Apply anisotropic diffusion filtering to reduce noise but preserve
    % sharp edges in CSF
    I = imdiffusefilt(nii(:,:,:,i), 'NumberOfIterations', 7);
    
    % Measure mean and std image intensity and create a probability map per
    % example label.
    for j = 1:4
        means(i,j) = nanmean(I(seg==j));
        stds(i,j) = nanstd(I(seg==j));
        probs(:,:,:,j) = pdf('Normal',I,means(i,j),stds(i,j));
    end
    
    % Segment image as maximum probability label
    [~, segmented_img] = max(probs, [], 4);
    
    % Ensure highest intenisity voxels are CSF
    segmented_img(I>means(i,4)) = 4;
    
    % Combine blood, white matter & gray matter into brain tissue. 
    segmented_img(segmented_img == 2) = 1;
    segmented_img(segmented_img == 3) = 1;
    segmented_img(segmented_img == 4) = 2;
    segmented_img(mask~=1) = 0;
    
    % Calculate fractional volumes per label
    volumes(i,1) = sum(segmented_img==1 & mask == 1, 'all')/sum(mask==1, 'all');
    volumes(i,2) = sum(segmented_img==2 & mask == 1, 'all')/sum(mask==1, 'all');
end
end

