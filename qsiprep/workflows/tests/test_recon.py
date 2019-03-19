''' Testing qsiprep workflows'''
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from qsiprep.workflows.recon.base import init_qsirecon_wf

test_data = Path("/Users/mcieslak/projects/qsiprep_output_data/noisefree")
tmp_path = Path("/Users/mcieslak/projects/qsiprep/scratch/testing/noisefree")


recon_spec = pkgrf('qsiprep', 'data/pipelines/3dshore_dsistudio_mrtrix.json')
work_dir = str(tmp_path.absolute() / "work")
output_dir = str(tmp_path.absolute() / "output")
bids_dir = str(test_data)

subject_list = ['abcd']
wf = init_qsirecon_wf(subject_list=subject_list,
                      name="dsistudio_test",
                      run_uuid="test",
                      work_dir=work_dir,
                      output_dir=output_dir,
                      bids_dir=bids_dir,
                      recon_input=bids_dir,
                      recon_spec=recon_spec,
                      low_mem=False,
                      omp_nthreads=1
                      )
wf.config['execution']['stop_on_first_crash'] = 'true'
wf.run()



from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from qsiprep.workflows.recon.base import init_qsirecon_wf
# Test mapmri
test_data = Path("/Users/mcieslak/projects/qsiprep_output_data/noisefree")
tmp_path = Path("/Users/mcieslak/projects/qsiprep/scratch/testing/noisefree")
recon_spec = pkgrf('qsiprep', 'data/pipelines/mapmri_dsistudio_mrtrix.json')
work_dir = str(tmp_path.absolute() / "work")
output_dir = str(tmp_path.absolute() / "output")
bids_dir = str(test_data)

subject_list = ['abcd']
wf = init_qsirecon_wf(subject_list=subject_list,
                      name="mapmri_test",
                      run_uuid="test",
                      work_dir=work_dir,
                      output_dir=output_dir,
                      bids_dir=bids_dir,
                      recon_input=bids_dir,
                      recon_spec=recon_spec,
                      low_mem=False,
                      omp_nthreads=1
                      )
wf.config['execution']['stop_on_first_crash'] = 'true'
wf.run()
