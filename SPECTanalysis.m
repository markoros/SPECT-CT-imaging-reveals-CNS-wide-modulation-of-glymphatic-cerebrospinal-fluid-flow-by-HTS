et = niftiread('_ac_et_stack.nii');
hdr = niftiinfo('_ac_et_stack.nii');
seg = niftiread('_Seg.nii');
efflux = niftiread('_efflux.nii.');
 
for i = 1:size(et,4)
 
    I = et(:,:,:,i);
 
    total_activity (i) = nansum(I(efflux == 8))*prod(hdr.PixelDimensions(1:3)/10);
    intracranial_activity(i) = nansum(I(seg == 1))*prod(hdr.PixelDimensions(1:3)/10);
    cervical_activity(i) = nansum(I(seg == 2))*prod(hdr.PixelDimensions(1:3)/10);
    thoracic_activity(i) = nansum(I(seg == 3))*prod(hdr.PixelDimensions(1:3)/10);
    lumbar_activity(i) = nansum(I(seg == 4))*prod(hdr.PixelDimensions(1:3)/10);
    kidney_activity(i) = nansum(I(seg == 6))*prod(hdr.PixelDimensions(1:3)/10);
    urine_activity(i) = nansum(I(seg == 7))*prod(hdr.PixelDimensions(1:3)/10);
    heart_activity(i) = nansum(I(seg == 8))*prod(hdr.PixelDimensions(1:3)/10);
end

total_activity = total_activity - total_activity(1);
 
intracranial_activity = intracranial_activity - intracranial_activity(1);
intracranial_activity = intracranial_activity / total_activity(4) *100;
 
cervical_activity = cervical_activity - cervical_activity(1);
cervical_activity = cervical_activity/total_activity(4)*100;
 
thoracic_activity = thoracic_activity - thoracic_activity(1);
thoracic_activity = thoracic_activity/total_activity(4)*100;
 
lumbar_activity = lumbar_activity - lumbar_activity(1);
lumbar_activity = lumbar_activity / total_activity(4) *100;
 
kidney_activity = kidney_activity - kidney_activity(1);
kidney_activity = kidney_activity / total_activity(4) *100;
 
urine_activity = urine_activity - urine_activity(1);
urine_activity = urine_activity / total_activity(4) *100;

heart_activity = heart_activity - heart_activity(1);
heart_activity = heart_activity/total_activity(4)*100;