import numpy as np
from pathlib import Path
import time
import os.path
import nibabel as nib
from glob import glob
import scipy.ndimage.filters
from nipype.interfaces.ants import ApplyTransforms, Registration
import tempfile

exps = [
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200608_A_LK_STR_FB_HTS_KD",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200610_A_LK_HTS_STR_KD_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200630_A_LK_HTS_KD_STR_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200701_B_LK_HTS_KD_STR_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200721_A_LK_STR_HTS",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200722_B_LK_STR_HTS_new",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200727_LK_STR_HTS",
    r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200728_B_LK_STR_HTS",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200609_A_LK_STR_FB_SALINE_KD",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200610_B_LK_SALINE_STR_KD_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200630_B_LK_SALINE_KD_STR_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200701_A_LK_SALINE_KD_STR_FB",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200721_B_LK_STR_SALINE",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200722_A_LK_STR_SALINE",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200727_B_LK_STR_SALINE",
    r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200728_A_LK_STR_SALINE"
]


def apply_transformation_to_image(input_file: Path, output_file: Path, transformation, reference: Path, invert=None, use_nn=False, transformation_list=False):
    transformer = ApplyTransforms(terminal_output='none')
    transformer.inputs.dimension = 3
    transformer.inputs.input_image = str(input_file)
    transformer.inputs.output_image = str(output_file)
    transformer.inputs.reference_image = str(reference)
    if transformation_list:
        transformer.inputs.transforms = transformation
    else:
        transformer.inputs.transforms = [str(transformation)]
    transformer.inputs.interpolation = 'Linear'
    if invert is not None:
        transformer.inputs.invert_transform_flags = invert
    if use_nn:
        transformer.inputs.interpolation = 'NearestNeighbor'
    transformer.run()


def apply_transformation_to_image_sequence(input_file: Path, output_file: Path, transformation: Path, reference: Path, tmpdir=None):
    with tempfile.TemporaryDirectory() as tmp:
        if tmpdir is not None:
            tmp = tmpdir
        image_sequence = nib.load(str(input_file))
        images = nib.four_to_three(image_sequence)
        for index, image in enumerate(images):
            tmp_input = Path(tmp, str(index).zfill(5)+'.nii')
            tmp_output = Path(tmp, str(index).zfill(5)+'_mc.nii')
            nib.save(image, str(tmp_input))
            apply_transformation_to_image(tmp_input, tmp_output, transformation, reference)
            os.remove(tmp_input)
        transformed_image_stack = nib.concat_images([nib.load(p) for p in sorted(glob(os.path.join(tmp, '*_mc.nii')))])
        transformed_image_stack.set_data_dtype(np.float32)
        nib.save(transformed_image_stack, str(output_file))


def named(file_or_dir_path, something_to_add_at_the_end, new_type=None, collect=None):
    path = os.path.dirname(file_or_dir_path)
    file_name = os.path.basename(file_or_dir_path)
    name_types = file_name.split('.', maxsplit=1)
    if len(name_types) > 1:
        name, types = name_types
        if new_type is None:
            new_path = os.path.join(path, f'{name}_{something_to_add_at_the_end}.{types}')
        else:
            new_path = os.path.join(path, f'{name}_{something_to_add_at_the_end}.{new_type}')
    else:
        name = name_types[0]
        new_path = os.path.join(path, os.path.join(name, something_to_add_at_the_end))
    if collect is not None:
        collect.append(new_path)
    return new_path


def nonlinear_registration():
    return Registration(
        dimension=3,
        transforms=['Rigid', 'Affine', 'SyN'],
        # convergence_threshold=[1e-8, 1e-8, 1e-8],
        metric=['MI', 'MI', 'MI'],
        metric_weight=[1] * 3,
        num_threads=7,
        number_of_iterations=[[500, 100, 50], [2000, 1000, 500], [2000, 2000, 2000, 1000, 500]],
        sampling_strategy=['Random', 'Random', 'Random'],
        sampling_percentage=[0.2, 0.2, 0.2],
        shrink_factors=[[16, 8, 2], [16, 8, 2], [16, 12, 10, 8, 2]],
        smoothing_sigmas=[[1, 1, 0], [1, 1, 0], [2, 2, 2, 1, 0]],
        sigma_units=['vox']*3,
        transform_parameters=[(0.50,), (2.0,), (0.25, 3.0, 0.0)],
        winsorize_lower_quantile=0.005,
        winsorize_upper_quantile=0.999,
        output_warped_image=True,
        use_histogram_matching=False,
        radius_or_number_of_bins=[32]*3,
        convergence_window_size=[20]*3,
        write_composite_transform=True,
        collapse_output_transforms=True,
        # initial_moving_transform_com=2,  # Align the moving_image nad fixed_image befor registration usingthe
        # geometric center of the images (=0), the image intensities (=1),or
        # the origin of the images (=2)
        terminal_output='file',
        # float=True,
        interpolation='BSpline'
    )


def register_and_stack_exps():
    new_affine = np.array([[0.3, 0., 0., 0],
                           [0., 0.3, 0., 0],
                           [0., 0., 0.3, 0],
                           [0., 0., 0., 1.]])
    tmp = r"E:\Marko_HTS\HTS_striatum_experiment\tmp"
    iteration = 2
    fixed_path = r"E:\Marko_HTS\HTS_striatum_experiment\stack_mean_i1.nii"
    for e in exps:
        moving_path = glob(os.path.join(e, '*_ct_stack_body.nii'))[0]
        new_moving_path = os.path.join(tmp, os.path.basename(moving_path))
        if not os.path.isfile(new_moving_path):
            moving_image = nib.load(moving_path)
            new_moving_image = nib.Nifti1Pair(moving_image.get_fdata().squeeze(), new_affine)
            nib.save(new_moving_image, new_moving_path)

        now = time.time()
        warped_path = named(moving_path, f'i{iteration}')
        if not os.path.isfile(warped_path):
            r = nonlinear_registration()
            r.inputs.fixed_image = fixed_path
            r.inputs.moving_image = new_moving_path
            r.inputs.output_transform_prefix = os.path.join(os.path.dirname(moving_path), 'tf_')
            r.inputs.output_warped_image = warped_path
            r.run()
        print('time this took:', (time.time()-now)/60, 'minutes')
    stack_all(iteration)


def stack_all(iteration):
    exps = glob(r"E:\Marko_HTS\HTS_striatum_experiment\*\*\*_ct_stack_body_i{}.nii".format(iteration))
    ims = [nib.load(e) for e in exps]
    # im = nib.concat_images(ims)
    # nib.save(im, r"E:\Marko_HTS\HTS_striatum_experiment\stack_all_i{}.nii".format(iteration))
    mean_im = np.mean([i.get_fdata() for i in ims], axis=0)
    mean_nii = nib.Nifti1Pair(mean_im, ims[0].affine, ims[0].header)
    nib.save(mean_nii, r"E:\Marko_HTS\HTS_striatum_experiment\stack_mean_i{}.nii".format(iteration))


def transform_et_stacks():
    new_affine = np.array([[0.3, 0., 0., 0],
                           [0., 0.3, 0., 0],
                           [0., 0., 0.3, 0],
                           [0., 0., 0., 1.]])
    tmp = r"E:\Marko_HTS\HTS_striatum_experiment\tmp"
    for e in exps:
        input_file = glob(os.path.join(e, '*_ac_et_stack.nii'))[0]
        output_file = named(input_file, 'w')
        if os.path.isfile(output_file):
            print('done:', output_file)
        else:
            new_input_file = os.path.join(tmp, os.path.basename(input_file))
            im = nib.load(input_file)
            new_im = nib.Nifti1Pair(im.get_fdata(), new_affine, im.header)
            new_im.set_data_dtype(np.float32)
            nib.save(new_im, new_input_file)
            transformation_file = os.path.join(os.path.dirname(input_file), 'tf_Composite.h5')
            reference = r"E:\Marko_HTS\HTS_striatum_experiment\stack_mean_i2.nii"
            apply_transformation_to_image_sequence(
                input_file=Path(new_input_file),
                output_file=output_file,
                transformation=Path(transformation_file),
                reference=Path(reference),
                tmpdir=tmp
            )


def average_time_series():
    exps_hts = [
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200608_A_LK_STR_FB_HTS_KD",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200610_A_LK_HTS_STR_KD_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200630_A_LK_HTS_KD_STR_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200701_B_LK_HTS_KD_STR_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200721_A_LK_STR_HTS",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200722_B_LK_STR_HTS_new",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200727_LK_STR_HTS",
        r"E:\Marko_HTS\HTS_striatum_experiment\HTS_group\20200728_B_LK_STR_HTS"
    ]
    exps_its = [
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200609_A_LK_STR_FB_SALINE_KD",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200610_B_LK_SALINE_STR_KD_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200630_B_LK_SALINE_KD_STR_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200701_A_LK_SALINE_KD_STR_FB",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200721_B_LK_STR_SALINE",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200722_A_LK_STR_SALINE",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200727_B_LK_STR_SALINE",
        r"E:\Marko_HTS\HTS_striatum_experiment\ITS_group\20200728_A_LK_STR_SALINE"
    ]
    output_dir = r"E:\Marko_HTS\HTS_striatum_experiment"
    for exps, name in zip([exps_hts, exps_its], ['hts', 'its']):
        paths = [glob(os.path.join(e, '*_et_stack_w_n.nii'))[0] for e in exps]
        matrix = None
        for p in paths:
            nii = nib.load(p)
            im = nii.get_fdata()
            if matrix is None:
                matrix = im/len(paths)
            else:
                matrix = matrix + im/len(paths)
        new_nii = nib.Nifti1Pair(matrix, nii.affine, nii.header)
        nib.save(new_nii, os.path.join(output_dir, name+'_mean_stack_n.nii'))


def normalize_et_sequence():
    needle = nib.load(r"E:\Marko_HTS\HTS_striatum_experiment\needle.nii.gz").get_fdata()
    for e in exps:
        path = glob(os.path.join(e, '*_et_stack_w.nii'))[0]
        output_file = named(path, 'n')
        if os.path.isfile(output_file):
            print('done:', output_file)
        else:
            nii = nib.load(path)
            stack = nib.four_to_three(nii)
            fourth = stack[3].get_fdata()
            first = stack[0].get_fdata()
            base = np.sum(first[needle == 0])
            norm = np.sum(fourth[needle == 0]) - base
            mean_base = np.mean(first[needle == 0])
            norm_stack = [nib.Nifti1Pair((frame.get_fdata() - mean_base)/norm * 100.0, frame.affine, frame.header) for frame in stack]
            new_nii = nib.concat_images(norm_stack)
            nib.save(new_nii, output_file)


def split_stack_to_frames():
    input_path = r"E:\Marko_HTS\HTS_striatum_experiment\hts_mean_stack_n.nii"
    output_directory = r"E:\Marko_HTS\HTS_striatum_experiment\hts_mean_frames"

    nii = nib.load(input_path)
    stack = nib.four_to_three(nii)
    for index, frame in enumerate(stack):
        output_path = os.path.join(output_directory, '{}_n.nii'.format(str(index).zfill(4)))
        nib.save(frame, output_path)


def subtract_frames():
    hts_directory = r"E:\Marko_HTS\HTS_striatum_experiment\hts_mean_frames"
    its_directory = r"E:\Marko_HTS\HTS_striatum_experiment\its_mean_frames"
    output_directory = r"E:\Marko_HTS\HTS_striatum_experiment\hts_minus_its_frames"
    cannula_path = r"E:\Marko_HTS\HTS_striatum_experiment\needle.nii.gz"
    cannula = nib.load(cannula_path).get_fdata()

    for i in range(22):
        hts_path = os.path.join(hts_directory, f'{str(i).zfill(4)}_n.nii')
        its_path = os.path.join(its_directory, f'{str(i).zfill(4)}_n.nii')
        hts_nii = nib.load(hts_path)
        its_nii = nib.load(its_path)
        hts_image = hts_nii.get_fdata() / 0.03**3
        hts_image[cannula == 1] = 0
        its_image = its_nii.get_fdata() / 0.03**3
        its_image[cannula == 1] = 0
        hts_image = scipy.ndimage.filters.gaussian_filter(hts_image, 3)
        its_image = scipy.ndimage.filters.gaussian_filter(its_image, 3)

        subtracted_image = hts_image - its_image
        subtracted_image[subtracted_image < 0] = 0
        subtracted_nii = nib.Nifti1Pair(subtracted_image, hts_nii.affine, hts_nii.header)
        output_path = os.path.join(output_directory, f'{str(i).zfill(4)}_hts_blur_positive.nii')
        nib.save(subtracted_nii, output_path)

        subtracted_image = its_image - hts_image
        subtracted_image[subtracted_image < 0] = 0
        subtracted_nii = nib.Nifti1Pair(subtracted_image, hts_nii.affine, hts_nii.header)
        output_path = os.path.join(output_directory, f'{str(i).zfill(4)}_its_blur_positive.nii')
        nib.save(subtracted_nii, output_path)


if __name__ == '__main__':
    subtract_frames()
