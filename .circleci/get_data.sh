# Edit these for project-wide testing
IMAGE=pennbbl/qsiprep:latest
PATCH='--patch-qsiprep /home/mcieslak/projects/qsiprep/qsiprep'
NTHREADS=8
WGET="wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q"


cat << DOC

Create input data for tests. A few files are automatically
created because they're used in all/most of the tests. 
Imaging data is only downloaded as needed based on the
second argument to the function.

Default data:
-------------

data/nipype.cfg
  Instructs nipype to stop on the first crash
data/eddy_config.json
  Configures eddy to perform few iterations so it 
  finishes quickly.
data/license.txt
  A freesurfer license file

DOC


get_config_data() {
    WORKDIR=$1
    DS=$2
    ENTRYDIR=`pwd`
    mkdir -p ${WORKDIR}/data
    cd ${WORKDIR}/data

    # Write the config file
    CFG=${WORKDIR}/data/nipype.cfg
    printf "[execution]\nstop_on_first_crash = true\n" > ${CFG}
    echo "poll_sleep_duration = 0.01" >> ${CFG}
    echo "hash_method = content" >> ${CFG}

    # Get an eddy config. It's used for some tests
    cat > ${WORKDIR}/data/eddy_config.json << "EOT"
{
  "flm": "linear",
  "slm": "linear",
  "fep": false,
  "interp": "spline",
  "nvoxhp": 100,
  "fudge_factor": 10,
  "dont_sep_offs_move": false,
  "dont_peas": false,
  "niter": 2,
  "method": "jac",
  "repol": true,
  "num_threads": 1,
  "is_shelled": true,
  "use_cuda": false,
  "cnr_maps": true,
  "residuals": false,
  "output_type": "NIFTI_GZ",
  "args": ""
}
EOT
    chmod a+r ${WORKDIR}/data/eddy_config.json

    # We always need a freesurfer license
    echo "cHJpbnRmICJtYXR0aGV3LmNpZXNsYWtAcHN5Y2gudWNzYi5lZHVcbjIwNzA2XG4gKkNmZVZkSDVVVDhyWVxuIEZTQllaLlVrZVRJQ3dcbiIgPiBsaWNlbnNlLnR4dAo=" | base64 -d | sh

    cd ${ENTRYDIR}
}


cat << DOC
DSDTI:
------

Downsampled DTI (single shell) data along with an EPI 
fieldmap. 

Contents:
^^^^^^^^^

 - data/DSDTI/dataset_description.json
 - data/DSDTI/README
 - data/DSDTI/sub-PNC
 - data/DSDTI/sub-PNC/anat
 - data/DSDTI/sub-PNC/anat/sub-PNC_T1w.json
 - data/DSDTI/sub-PNC/anat/sub-PNC_T1w.nii.gz
 - data/DSDTI/sub-PNC/dwi
 - data/DSDTI/sub-PNC/dwi/sub-PNC_acq-realistic_dwi.bval
 - data/DSDTI/sub-PNC/dwi/sub-PNC_acq-realistic_dwi.bvec
 - data/DSDTI/sub-PNC/dwi/sub-PNC_acq-realistic_dwi.json
 - data/DSDTI/sub-PNC/dwi/sub-PNC_acq-realistic_dwi.nii.gz
 - data/DSDTI/sub-PNC/fmap
 - data/DSDTI/sub-PNC/fmap/sub-PNC_dir-PA_epi.json
 - data/DSDTI/sub-PNC/fmap/sub-PNC_dir-PA_epi.nii.gz


DSCSDSI:
--------

Downsampled CS-DSI data. 

Contents:
^^^^^^^^^
 - data/DSCSDSI_nofmap/dataset_description.json
 - data/DSCSDSI_nofmap/README
 - data/DSCSDSI_nofmap/sub-tester
 - data/DSCSDSI_nofmap/sub-tester/anat
 - data/DSCSDSI_nofmap/sub-tester/anat/sub-tester_T1w.json
 - data/DSCSDSI_nofmap/sub-tester/anat/sub-tester_T1w.nii.gz
 - data/DSCSDSI_nofmap/sub-tester/dwi
 - data/DSCSDSI_nofmap/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.bval
 - data/DSCSDSI_nofmap/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.bvec
 - data/DSCSDSI_nofmap/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.json
 - data/DSCSDSI_nofmap/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.nii.gz


DSCSDSI_BUDS:
-------------

Downsampled CS-DSI data with blip-up and blip-down DWI series.

Contents:
^^^^^^^^^

 - data/DSCSDSI_BUDS
 - data/DSCSDSI_BUDS/dataset_description.json
 - data/DSCSDSI_BUDS/README
 - data/DSCSDSI_BUDS/sub-tester
 - data/DSCSDSI_BUDS/sub-tester/anat
 - data/DSCSDSI_BUDS/sub-tester/anat/sub-tester_T1w.json
 - data/DSCSDSI_BUDS/sub-tester/anat/sub-tester_T1w.nii.gz
 - data/DSCSDSI_BUDS/sub-tester/dwi
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.bval
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.bvec
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.json
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55AP_dwi.nii.gz
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55PA_dwi.bval
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55PA_dwi.bvec
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55PA_dwi.json
 - data/DSCSDSI_BUDS/sub-tester/dwi/sub-tester_acq-HASC55PA_dwi.nii.gz
 - data/DSCSDSI_BUDS/sub-tester/fmap
 - data/DSCSDSI_BUDS/sub-tester/fmap/sub-tester_dir-AP_epi.json
 - data/DSCSDSI_BUDS/sub-tester/fmap/sub-tester_dir-AP_epi.nii.gz
 - data/DSCSDSI_BUDS/sub-tester/fmap/sub-tester_dir-PA_epi.json
 - data/DSCSDSI_BUDS/sub-tester/fmap/sub-tester_dir-PA_epi.nii.gz


twoses:
-------

Data containing two sessions.

Contents:
^^^^^^^^^

 - data/twoses/dataset_description.json
 - data/twoses/README
 - data/twoses/sub-tester/ses-1/anat/sub-tester_ses-1_T1w.json
 - data/twoses/sub-tester/ses-1/anat/sub-tester_ses-1_T1w.nii.gz
 - data/twoses/sub-tester/ses-1/dwi
 - data/twoses/sub-tester/ses-1/dwi/sub-tester_ses-1_acq-HASC55PA_dwi.bval
 - data/twoses/sub-tester/ses-1/dwi/sub-tester_ses-1_acq-HASC55PA_dwi.bvec
 - data/twoses/sub-tester/ses-1/dwi/sub-tester_ses-1_acq-HASC55PA_dwi.json
 - data/twoses/sub-tester/ses-1/dwi/sub-tester_ses-1_acq-HASC55PA_dwi.nii.gz
 - data/twoses/sub-tester/ses-2
 - data/twoses/sub-tester/ses-2/dwi
 - data/twoses/sub-tester/ses-2/dwi/sub-tester_ses-2_acq-HASC55AP_dwi.bval
 - data/twoses/sub-tester/ses-2/dwi/sub-tester_ses-2_acq-HASC55AP_dwi.bvec
 - data/twoses/sub-tester/ses-2/dwi/sub-tester_ses-2_acq-HASC55AP_dwi.json
 - data/twoses/sub-tester/ses-2/dwi/sub-tester_ses-2_acq-HASC55AP_dwi.nii.gz


multishell_output:
------------------

Results from running qsiprep on a simulated ABCD (multi-shell) dataset

Contents:
^^^^^^^^^

 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_desc-brain_mask.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_desc-preproc_T1w.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_dseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_from-orig_to-T1w_mode-image_xfm.txt
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_label-CSF_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_label-GM_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_label-WM_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_dseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/anat/sub-ABCD_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_confounds.tsv
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_b0series.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-brain_mask.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-eddy_cnr.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi.b
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi.bval
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi.bvec
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi.nii.gz
 - data/multishell_output/qsiprep/sub-ABCD/dwi/sub-ABCD_acq-10per000_space-T1w_dwiref.nii.gz


singleshell_output:
-------------------

Preprocessed data from a single-shell dataset

Contents:
^^^^^^^^^
 - data/singleshell_output/qsiprep/dataset_description.json
 - data/singleshell_output/qsiprep/logs/CITATION.html
 - data/singleshell_output/qsiprep/logs/CITATION.md
 - data/singleshell_output/qsiprep/logs/CITATION.tex
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_desc-brain_mask.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_desc-preproc_T1w.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_dseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_from-orig_to-T1w_mode-image_xfm.txt
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_label-CSF_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_label-GM_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_label-WM_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_dseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/anat/sub-PNC_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_confounds.tsv
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_b0series.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-brain_mask.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-eddy_cnr.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-preproc_dwi.b
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-preproc_dwi.bval
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-preproc_dwi.bvec
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-preproc_dwi.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_dwiref.nii.gz
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_acq-realistic_carpetplot.svg
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_acq-realistic_coreg.svg
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_acq-realistic_desc-sdc_b0.svg
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_acq-realistic_sampling_scheme.gif
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_seg_brainmask.svg
 - data/singleshell_output/qsiprep/sub-PNC/figures/sub-PNC_t1_2_mni.svg
 - data/singleshell_output/qsiprep/sub-PNC.html

DOC


get_bids_data() {
    WORKDIR=$1
    DS=$2
    ENTRYDIR=`pwd`
    mkdir -p ${WORKDIR}/data
    cd ${WORKDIR}/data

    # Down-sampled compressed sensing DSI
    if [[ ${DS} = DSCSDSI ]]; then
      ${WGET} \
        -O DSCSDSI_nofmap.tar.xz \
	    "https://upenn.box.com/shared/static/eq6nvnyazi2zlt63uowqd0zhnlh6z4yv.xz"
      tar xvfJ DSCSDSI_nofmap.tar.xz -C ${WORKDIR}/data/
      rm DSCSDSI_nofmap.tar.xz
    fi

    # Get BUDS scans from downsampled CS-DSI
    if [[ ${DS} = DSCSDSI_BUDS ]]; then
      ${WGET} \
        -O dscsdsi_buds.tar.xz \
        "https://upenn.box.com/shared/static/bvhs3sw2swdkdyekpjhnrhvz89x3k87t.xz"
      tar xvfJ dscsdsi_buds.tar.xz -C ${WORKDIR}/data/
      rm dscsdsi_buds.tar.xz
    fi

    # Get downsampled DTI
    if [[ ${DS} = DSDTI ]]; then
		${WGET} \
        -O DSDTI.tar.xz \
        "https://upenn.box.com/shared/static/iefjtvfez0c2oug0g1a9ulozqe5il5xy.xz"
      tar xvfJ DSDTI.tar.xz -C ${WORKDIR}/data/
      rm DSDTI.tar.xz
    fi

    # Get multisession CS-DSI
    if [[ ${DS} = twoses ]]; then
      ${WGET} \
        -O twoses.tar.xz \
        "https://upenn.box.com/shared/static/c949fjjhhen3ihgnzhkdw5jympm327pp.xz"
      tar xvfJ twoses.tar.xz -C ${WORKDIR}/data/
      rm twoses.tar.xz
    fi

    # Get Multi Shell outputs
    if [[ ${DS} = multishell_output ]]; then
      ${WGET} \
        -O multishell_output.tar.gz \
        "https://upenn.box.com/shared/static/nwxdn4ale8dkebvpjmxbx99dqjzwvlmh.gz"
      tar xvfz multishell_output.tar.gz -C ${WORKDIR}/data/
      rm multishell_output.tar.gz
    fi

    #  name: Get Single Shell outputs
    if [[ ${DS} = singleshell_output ]]; then
      mkdir -p ${WORKDIR}/data/singleshell_output
      ${WGET} \
        -O singleshell_output.tar.gz \
        "https://upenn.box.com/shared/static/9jhf0eo3ml6ojrlxlz6lej09ny12efgg.gz"
      tar xvfz singleshell_output.tar.gz -C ${WORKDIR}/data/singleshell_output
      rm singleshell_output.tar.gz
    fi
    cd ${ENTRYDIR}
}

setup_dir(){
    # Create the output and working directories for 
    WORKDIR=$1
    DS=$2
    mkdir -p ${WORKDIR}/${DS}/work ${WORKDIR}/${DS}/derivatives
    setfacl -d -m group:$(id -gn):rwx ${WORKDIR}/${DS}/derivatives && \
        setfacl -m group:$(id -gn):rwx ${WORKDIR}/${DS}/derivatives
    setfacl -d -m group:$(id -gn):rwx ${WORKDIR}/${DS}/work && \
        setfacl -m group:$(id -gn):rwx ${WORKDIR}/${DS}/work

}

