import unittest
import tempfile
import os
from quantum_tomography import measurements, tomography, utils


class TomographyIntegrationTest(unittest.TestCase):
    def test_reconstruction_no_noise(self):
        with tempfile.TemporaryDirectory() as td:
            # Simulate a very small measurement campaign with no noise
            measurements.simulate_measurements(
                state_type='GHZ',
                N=2,
                symmetry='Permutational',
                c=3,
                shots=[1000],
                noise_types=['Depolarizing'],
                noise_list=[0.0],
                save_root=os.path.join(td, 'Measurements'),
            )

            meas_root = os.path.join(td, 'Measurements')
            # There should be a folder Measurements/GHZ_2qubits_Permutational
            folders = os.listdir(meas_root)
            self.assertTrue(len(folders) > 0)
            measurements_folder = os.path.join(meas_root, folders[0])

            results_folder = os.path.join(td, 'Results')
            tomography.run_tomography(measurements_folder, results_folder)

            # Check results exist
            res_dirs = os.listdir(results_folder)
            self.assertTrue(len(res_dirs) > 0)
            noise_dir = os.path.join(results_folder, res_dirs[0], 'Depolarizing')
            # Load one of the reconstructed files
            files = os.listdir(noise_dir)
            # Find DMtomo file
            dm_files = [f for f in files if f.startswith('DMtomo')]
            self.assertTrue(len(dm_files) > 0)
            dm_path = os.path.join(noise_dir, dm_files[0])
            RMs = utils.load_np(dm_path)
            # Compare first reconstructed RM against ideal
            DM_ideal = utils.load_np(os.path.join(measurements_folder, 'Depolarizing', 'DM_ideal.npy'))
            # RMs is an array of reconstructed matrices
            RM0 = RMs[0]
            fid = utils.fidelity(DM_ideal, RM0)
            self.assertGreater(fid, 0.98)


if __name__ == '__main__':
    unittest.main()
