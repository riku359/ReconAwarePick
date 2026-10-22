"""Small CPU regressions; run with PYTHONPATH=src python -m unittest discover -s tests -p test_validation.py."""
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEANER = ROOT / 'src/rapick/cleaner/filter_star_from_masks.py'
HEADER = 'data_particles\n\nloop_\n_rlnMicrographName #1\n_rlnCoordinateX #2\n_rlnCoordinateY #3\n'


class MaskFiltering(unittest.TestCase):
    def setUp(self):
        # Retain fixtures for inspection and validation evidence.
        self.root = Path(tempfile.mkdtemp(prefix='rapick-regression-'))
        self.star = self.root / 'picks.star'
        self.star.write_text(HEADER + 'mic.mrc 0 0\nmic.mrc 1 1\nmic.mrc 2 2\n')
        self.out = self.root / 'out'
        self.out.mkdir()

    def run_filter(self, *extra):
        return subprocess.run([sys.executable, str(CLEANER), '--star', str(self.star),
                               '--mask-dir', str(self.root), '--out-dir', str(self.out),
                               '--empiar-id', '10081', *extra], capture_output=True, text=True)

    def mask(self, values):
        np.savez(self.root / 'mic_tri.npz', tri=np.array(values, dtype=np.float16),
                 meta=json.dumps({'width': 3, 'height': 3}))

    def test_missing_preserves_outputs_and_decisions(self):
        target = self.out / 'cryotransformer_clean_tri.star'
        target.write_text('previous valid result\n')
        decisions = self.out / 'decisions_tri.jsonl'
        decisions.write_text('previous decisions\n')
        result = self.run_filter('--overwrite')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('required mask', result.stderr)
        self.assertEqual(target.read_text(), 'previous valid result\n')
        self.assertEqual(decisions.read_text(), 'previous decisions\n')

    def test_corrupt_and_nonfinite_fail(self):
        (self.root / 'mic_tri.npz').write_bytes(b'invalid archive')
        self.assertNotEqual(self.run_filter().returncode, 0)
        self.mask([[0, 0, 0], [0, float('nan'), 0], [0, 0, 0]])
        self.assertNotEqual(self.run_filter().returncode, 0)

    def test_threshold_coordinates_and_resume(self):
        # STAR Y is bottom-up: the stored mask is flipped. Exactly 0.5 is removed.
        self.mask([[1, 1, 0.499], [1, 0.5, 1], [0.0, 1, 1]])
        result = self.run_filter()
        self.assertEqual(result.returncode, 0, result.stderr)
        text = (self.out / 'cryotransformer_clean_tri.star').read_text()
        self.assertIn('mic.mrc 0 0', text)
        self.assertIn('mic.mrc 2 2', text)
        self.assertNotIn('mic.mrc 1 1', text)
        self.assertEqual(self.run_filter().returncode, 0)
        # A successful cached decision must not conceal a newly corrupted mask.
        (self.root / 'mic_tri.npz').write_bytes(b'corrupt after success')
        self.assertNotEqual(self.run_filter().returncode, 0)
        self.assertEqual((self.out / 'cryotransformer_clean_tri.star').read_text(), text)

    def test_decisions_recompute_when_threshold_changes(self):
        self.mask([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertEqual(self.run_filter().returncode, 0)
        self.assertEqual(json.loads((self.out / 'summary_tri.json').read_text())['picks_kept'], 0)
        self.assertEqual(self.run_filter('--deep-thr', '1.1').returncode, 0)
        self.assertEqual(json.loads((self.out / 'summary_tri.json').read_text())['picks_kept'], 3)

    def test_explicit_permissive_records_errors(self):
        result = self.run_filter('--allow-missing-masks')
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.out / 'summary_tri.json').read_text())
        self.assertEqual(summary['n_micrographs_error'], 1)
        self.assertEqual(summary['picks_kept'], 3)


class Teacher(unittest.TestCase):
    def test_prefixed_gt_and_empty_image(self):
        from rapick.loop.make_gt_teacher import build_teacher
        with nullcontext(tempfile.mkdtemp(prefix='rapick-teacher-')) as tmp:
            p = Path(tmp)
            (p / 'gt.star').write_text(HEADER + 'J1/imported/12345_mic.mrc 17 29\nother.mrc 8 9\n')
            (p / 'mics.txt').write_text('mic.mrc\nempty\n')
            summary = build_teacher('10081', p / 'mics.txt', p / 'teacher', p / 'gt.star')
            self.assertEqual(summary['n_teacher_particles'], 1)
            self.assertEqual(summary['dropped_empty_gt_mics'], ['empty'])
            self.assertIn('12345_mic.mrc 17 29', (p / 'teacher/teacher_gt.star').read_text())


if __name__ == '__main__':
    unittest.main()


class SavedParticles(unittest.TestCase):
    def test_coordinate_inverse_and_seeded_export_without_credentials(self):
        from unittest.mock import patch
        from rapick.loop import export_teacher_star as exporter
        p = Path(tempfile.mkdtemp(prefix='rapick-particles-'))
        dtype = [('location/micrograph_path', 'S80'),
                 ('location/micrograph_shape', 'i4', (2,)),
                 ('location/center_x_frac', 'f4'), ('location/center_y_frac', 'f4')]
        rows = np.zeros(3, dtype=dtype)
        rows['location/micrograph_path'] = [b'J1/123_mic.mrc', b'J1/456_second.mrc', b'J1/789_third.mrc']
        rows['location/micrograph_shape'] = (100, 200)
        rows['location/center_x_frac'] = .25
        rows['location/center_y_frac'] = .2
        keys, xs, ys = exporter.particle_coords(rows)
        self.assertEqual(keys, ['mic', 'second', 'third'])
        np.testing.assert_allclose(xs, [50, 50, 50])
        np.testing.assert_allclose(ys, [80, 80, 80])
        with (p / 'selected.cs').open('wb') as f:
            np.save(f, rows)
        outputs = []
        with patch.object(exporter, 'connect_cryosparc', side_effect=AssertionError('offline connected')):
            for n in [1, 2]:
                out = p / str(n)
                self.assertEqual(exporter.main(['--particles-cs', str(p / 'selected.cs'),
                                 '--empiar', '10081', '--seed', '1', '--num-mics', '2',
                                 '--out-dir', str(out)]), 0)
                outputs.append((out / 'teacher.star').read_text())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn('50 80', outputs[0])


class OfflineScoring(unittest.TestCase):
    def test_job_directory_does_not_read_credentials(self):
        from unittest.mock import patch
        from rapick.select2d import score_class2d as scorer
        meta = dict(n_classes=2, n_particles=np.array([10, 20]), n_particles_total=30)
        with patch.object(scorer, 'read_env', side_effect=AssertionError('read credentials')), \
             patch.object(scorer, 'score_job', return_value=(np.array([3.499, 3.5]), meta, 'scores.csv')), \
             patch.object(sys, 'argv', ['score', '--job', 'J15', '--job-dir', 'saved', '--out-root', 'out']):
            scorer.main()


class ClassRules(unittest.TestCase):
    def test_class_quantile_ties_and_box_boundaries(self):
        from rapick.select2d.iterate_class2d import attractor_threshold, plan_for_box
        self.assertEqual(attractor_threshold([1., 2., 2., 2.5, 4.9], 2.5, .7), 2.)
        self.assertEqual(attractor_threshold([], 2.5, .7), 2.5)
        self.assertEqual(plan_for_box(199), ('class_2D_new', 5))
        self.assertEqual(plan_for_box(200), ('class_2D', 3))
        self.assertEqual(plan_for_box(300), ('class_2D', 3))
        self.assertEqual(plan_for_box(301), ('class_2D', 2))

    def test_selection_excludes_threshold_and_counts_particles(self):
        from rapick.select2d.cryosift_jobs import select_classes
        class SavedJob:
            def interact(self, name, payload=None):
                if name == 'get_class_info':
                    return [{'class_idx': i, 'num_particles_total': n}
                            for i, n in enumerate([100, 1, 500])]
        result = select_classes(SavedJob(), {0: 3.499, 1: 3.5, 2: 4.5}, lambda score: score < 3.5)
        self.assertEqual(result.kept_classes, [0])
        self.assertEqual(result.kept_particles, 100)
        self.assertEqual(result.dropped_particles, 501)


class MicrographNames(unittest.TestCase):
    def test_numeric_dates_survive_import_prefix_matching(self):
        from rapick.loop.filter_star_by_micrograph_list import listed_micrograph
        self.assertTrue(listed_micrograph('123_20210903_image.mrc', {'20210903_image'}))
        self.assertTrue(listed_micrograph('20210903_image.mrc', {'20210903_image'}))
        self.assertFalse(listed_micrograph('20210904_image.mrc', {'20210903_image'}))


class RoundZeroGate(unittest.TestCase):
    def test_metric_drift_stops_before_marking_score(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from rapick.loop import run_loop
        gate = dict(macro_P=.5, macro_R=.8, macro_F1=.6, n_pred_eval=100)
        ctx = SimpleNamespace(empiar='10081', entry=SimpleNamespace(gate=gate, gt_star=Path('gt.star')))
        state = Mock()
        payload = dict(gate, macro_P=.7)
        with patch.object(run_loop.paths, 'tool_script', return_value=Path('scorer.py')), \
             patch.object(run_loop.paths, 'tool_cmd', return_value=['python', 'scorer.py']), \
             patch.object(run_loop, 'run', return_value='JSON ' + json.dumps(payload)):
            with self.assertRaisesRegex(RuntimeError, 'Selection is legacy_idxfix'):
                run_loop.step_score(0, state, Path('round0'), ctx)
        state.mark.assert_not_called()


class RepickCommand(unittest.TestCase):
    def test_fullset_uses_selection_aware_predictor(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from rapick.loop import repick_fullset, paths
        root = Path(tempfile.mkdtemp(prefix='rapick-command-'))
        checkpoint = root / 'fixture.pth'
        checkpoint.touch()
        args = SimpleNamespace(model=checkpoint, gpu=0, condition='fixture')
        entry = SimpleNamespace(empiar='10081')
        # Stop exactly at the process boundary, before any neural computation.
        with patch.object(paths, 'fullset_data_root', return_value=root), \
             patch.object(paths, 'picker_images', return_value=root), \
             patch.object(repick_fullset, 'run', side_effect=InterruptedError) as launch:
            with self.assertRaises(InterruptedError):
                repick_fullset.step_pick(Mock(), root, args, entry)
        command = launch.call_args.args[0]
        self.assertEqual(Path(command[1]).name, 'predict.py')
        self.assertEqual(command[command.index('--selection') + 1], 'legacy_idxfix')
        python = ROOT / 'envs/cryotransformer/.venv/bin/python'
        if python.exists():
            # Parse the actual constructed command with the installed overlay,
            # without allocating a model or requiring a GPU.
            code = ('import predict,sys; '
                    'predict.get_args_parser().parse_args(sys.argv[1:])')
            result = subprocess.run([str(python), '-c', code, *command[2:]],
                                    cwd=Path(command[1]).parent,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
