"""Unit-safe bandwidth accounting and retained Cube counting contracts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'assets/skills/ascend-roofline/scripts'
sys.path.insert(0, str(SCRIPTS))
import calculate_mfu as mfu
import calculate_roofline as roofline


class BandwidthTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / 'memory.csv'
        self.csv.write_text('read,write,transfer\n64,68.5,66\n999,999,999\n', encoding='utf-8')

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPTS / 'calculate_bwu.py'),
                               '--csv', str(self.csv), '--resource', 'HBM',
                               '--direction', 'read+write', '--time-us', '2.88',
                               '--time-source', 'synthetic task fixture', '--time-scope', 'task',
                               *extra], capture_output=True, text=True)

    def test_select_rows_and_columns_convert_kib_to_si_without_double_count(self):
        result = self.run_cli('--row', '1', '--column', 'read', '--column', 'write',
                              '--amount-unit', 'KiB', '--peak-gbps', '1600',
                              '--peak-source', 'hypothetical shared read/write peak')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['total_bytes'], 135680)
        self.assertAlmostEqual(data['achieved_gbps'], 47.1111111111)
        self.assertAlmostEqual(data['bwu_percent'], 2.94444444444)
        self.assertEqual(data['time_scope'], 'task')

    def test_no_peak_gives_bandwidth_without_inventing_utilization(self):
        result = self.run_cli('--row', '1', '--column', 'read', '--amount-unit', 'KB')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['total_bytes'], 64000)
        self.assertIsNone(data['bwu_percent'])

    def test_missing_na_and_nonfinite_values_are_not_zero(self):
        for value in ('NA', '', 'nan', 'inf', '-1'):
            with self.subTest(value=value):
                self.csv.write_text(f'read,write\n{value},1\n', encoding='utf-8')
                result = self.run_cli('--row', '1', '--column', 'read', '--amount-unit', 'KiB')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('row 1', result.stderr)

    def test_ambiguous_selection_and_unknown_peak_source_are_rejected(self):
        for extra in (
            ['--column', 'read'],
            ['--row', '1', '--row', '1', '--column', 'read'],
            ['--row', '1', '--column', 'read', '--column', 'read'],
            ['--row', '3', '--column', 'read'],
            ['--row', '1', '--column', 'absent'],
            ['--row', '1', '--column', 'read', '--peak-gbps', '1600'],
        ):
            with self.subTest(extra=extra):
                result = self.run_cli('--amount-unit', 'KiB', *extra)
                self.assertNotEqual(result.returncode, 0)

    def test_zero_copy_payload_and_above_peak_are_preserved(self):
        self.csv.write_text('read\n0\n100\n', encoding='utf-8')
        for row, expected in [('1', 0), ('2', 3472.2222222222)]:
            result = self.run_cli('--row', row, '--column', 'read', '--amount-unit', 'KB',
                                  '--peak-gbps', '1', '--peak-source', 'synthetic')
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertAlmostEqual(data['bwu_percent'], expected)
            self.assertEqual(bool(data['warnings']), row == '2')


class CubeCompatibilityTest(unittest.TestCase):
    def test_causal_attention_masks_reduction_separately(self):
        qk = mfu.gemm_ops(dict(m=4, n=4, k=2, mask='causal'))
        pv = mfu.gemm_ops(dict(m=4, n=3, k=4, valid_macs=30))
        self.assertEqual(qk['useful_ops'] + pv['useful_ops'], 100)

    def test_bundled_model_keeps_useful_and_executed_distinct(self):
        spec = json.loads((SCRIPTS.parent / 'assets/roofline-example.json').read_text())
        result = roofline.calculate(spec)
        self.assertGreater(spec['executed_ops'], spec['useful_ops'])
        self.assertLess(result['mfu_ceiling_percent'], 100)


class GeneralModelTest(unittest.TestCase):
    def test_legacy_cube_mfu_keeps_catalog_peak_and_counts(self):
        data=mfu.calculate('950pr',32,32,'bf16',1,'ms',1,[dict(m=4096,n=4096,k=4096)])
        self.assertEqual(data['used_cube_peak'],432)
        self.assertAlmostEqual(data['mfu_percent'],2*4096**3/(1e-3*432e12)*100)

    def test_integer_roofline_does_not_label_utilization_mfu(self):
        spec=dict(engine='vector',device='950pr',device_cores=28,cores=56,dtype='int32',
                  useful_ops=11000000,executed_ops=11000000,resources=[],
                  assumptions=['synthetic integer work'],measured_time_us=2)
        data=roofline.calculate(spec)
        self.assertEqual(data['peak_unit'],'TOPS')
        self.assertIsNone(data['measured_mfu_percent'])
        self.assertAlmostEqual(data['measured_utilization_percent'],50)

    def test_explicit_peak_rejects_mixed_catalog_selection(self):
        spec=dict(engine='vector',dtype='fp32',device='950pr',peak_tops=23,
                  peak_source='test',useful_ops=1,executed_ops=1,resources=[],assumptions=['test'])
        with self.assertRaisesRegex(ValueError,'mutually exclusive'): roofline.calculate(spec)

    def test_examples_and_reference_structure(self):
        ref=SCRIPTS.parent/'references'
        self.assertEqual({p.name for p in ref.iterdir()},
                         {'peaks.md','counts.md','roofline.md','mfu.md','bwu.md','hardware.md'})
        for name in ('roofline-example.json','vector-example.json'):
            model=json.loads((SCRIPTS.parent/'assets'/name).read_text())
            self.assertGreater(roofline.calculate(model)['time_floor_us'],0)
        hardware=json.loads((SCRIPTS.parent/'assets/hardware.json').read_text())
        self.assertTrue(hardware['variants'])
        for variant in hardware['variants']:
            self.assertIsNone(variant['verified_l2_bandwidth_gbps'])

    def test_vector_catalog_and_mfu(self):
        result = subprocess.run([sys.executable, str(SCRIPTS / 'calculate_mfu.py'),
            '--engine', 'vector', '--device', '950pr', '--device-cores', '28',
            '--cores', '56', '--dtype', 'fp32', '--useful-ops', '23000000',
            '--time', '2', '--unit', 'us', '--json'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data=json.loads(result.stdout)
        self.assertAlmostEqual(data['mfu_percent'], 50)
        self.assertEqual(data['used_peak'], 23)

    def test_vector_roofline_with_explicit_matching_peak(self):
        spec=dict(engine='vector', dtype='fp32', peak_tops=10,
                  peak_source='synthetic FMA resource', useful_ops=1000000,
                  executed_ops=2000000, resources=[dict(name='main-memory',
                  payload_bytes={'input':1000000}, bandwidth_gbps=1000,
                  basis='assumed', source='synthetic bandwidth')],
                  assumptions=['synthetic fixture; ideal overlap'], measured_time_us=2)
        result=roofline.calculate(spec)
        self.assertAlmostEqual(result['time_floor_us'],1)
        self.assertAlmostEqual(result['mfu_ceiling_percent'],10)
        self.assertAlmostEqual(result['measured_mfu_percent'],5)
        self.assertAlmostEqual(result['resources'][0]['bwu_percent'],50)
        self.assertEqual(result['limiting_resources'],['main-memory'])

    def test_catalog_vector_core_count_is_aiv_not_aic(self):
        spec=dict(engine='vector', device='950dt', device_cores=36, cores=72,
                  dtype='fp32', useful_ops=30000000, executed_ops=30000000,
                  resources=[], assumptions=['catalog nominal peak'])
        result=roofline.calculate(spec)
        self.assertEqual(result['used_peak_tops'],30)
        self.assertAlmostEqual(result['time_floor_us'],1)
        spec['cores']=73
        with self.assertRaises(ValueError): roofline.calculate(spec)

    def test_unsupported_vector_dtype_and_unverified_explicit_peak(self):
        spec=dict(engine='vector', device='950pr', device_cores=28, cores=56,
                  dtype='tf32', useful_ops=1, executed_ops=1,
                  resources=[], assumptions=['test'])
        with self.assertRaises(ValueError): roofline.calculate(spec)
        spec=dict(engine='vector',dtype='fp32',peak_tops=10,useful_ops=1,
                  executed_ops=1,resources=[],assumptions=['test'])
        with self.assertRaises(ValueError): roofline.calculate(spec)

    def test_pure_copy_has_bwu_without_fictitious_mfu(self):
        spec=dict(engine='vector',dtype='fp32',peak_tops=10,peak_source='unused for copy',
                  useful_ops=0,executed_ops=0,resources=[dict(name='memory',
                  payload_bytes={'rw':1000000},bandwidth_gbps=1000,
                  basis='assumed',source='synthetic')], assumptions=['copy'],measured_time_us=2)
        result=roofline.calculate(spec)
        self.assertIsNone(result['measured_mfu_percent'])
        self.assertIsNone(result['measured_utilization_percent'])
        self.assertIsNone(result['utilization_ceiling_percent'])
        self.assertEqual(result['resources'][0]['bwu_percent'],50)

    def test_fully_wasted_compute_has_zero_not_missing_mfu(self):
        spec=dict(engine='vector',dtype='fp32',peak_tops=10,peak_source='synthetic',
                  useful_ops=0,executed_ops=1000,resources=[],assumptions=['all work masked'],
                  measured_time_us=1)
        result=roofline.calculate(spec)
        self.assertEqual(result['measured_mfu_percent'],0)
        self.assertEqual(result['measured_utilization_percent'],0)
        self.assertEqual(result['utilization_ceiling_percent'],0)

    def test_integer_model_reports_utilization_ceiling_without_calling_it_mfu(self):
        spec=dict(engine='vector',device='950pr',device_cores=28,cores=56,dtype='int32',
                  useful_ops=11000000,executed_ops=11000000,resources=[],
                  assumptions=['synthetic integer work'])
        result=roofline.calculate(spec)
        self.assertAlmostEqual(result['utilization_ceiling_percent'],100)
        self.assertIsNone(result['mfu_ceiling_percent'])

    def test_multiple_resources_take_the_max_floor_and_name_the_limiter(self):
        spec=dict(engine='cube',dtype='fp32',peak_tops=100,peak_source='synthetic',
                  useful_ops=1000000,executed_ops=1000000,resources=[
                      dict(name='main-memory',payload_bytes={'io':1000000},bandwidth_gbps=1000,
                           basis='assumed',source='synthetic'),
                      dict(name='l2',payload_bytes={'io':1000000},bandwidth_gbps=10000,
                           basis='assumed',source='synthetic')],
                  assumptions=['independent resources; ideal overlap'])
        result=roofline.calculate(spec)
        self.assertAlmostEqual(result['time_floor_us'],1)
        self.assertEqual(result['limiting_resources'],['main-memory'])
        self.assertNotIn('bwu_percent',result['resources'][0])

    def test_zero_payload_resource_keeps_ops_per_byte_unset(self):
        spec=dict(engine='cube',dtype='fp32',peak_tops=10,peak_source='synthetic',
                  useful_ops=1000,executed_ops=1000,resources=[dict(name='memory',
                  payload_bytes={'none':0},bandwidth_gbps=1000,basis='assumed',
                  source='synthetic')],assumptions=['zero traffic'])
        result=roofline.calculate(spec)
        self.assertIsNone(result['resources'][0]['useful_ops_per_byte'])
        self.assertEqual(result['limiting_resources'],['cube'])

    def test_explicit_peak_ignores_null_catalog_fields(self):
        spec=dict(engine='cube',device=None,device_cores=None,cores=None,dtype='bf16',
                  peak_tops=10,peak_source='synthetic',useful_ops=1,executed_ops=1,
                  resources=[],assumptions=['explicit peak'])
        self.assertAlmostEqual(roofline.calculate(spec)['used_peak_tops'],10)

    def test_incomplete_catalog_selection_names_the_missing_fields(self):
        spec=dict(engine='cube',device='950pr',dtype='bf16',useful_ops=1,executed_ops=1,
                  resources=[],assumptions=['missing cores'])
        with self.assertRaisesRegex(ValueError,'provide device'): roofline.calculate(spec)

    def test_peak_source_without_peak_tops_is_rejected(self):
        spec=dict(engine='cube',dtype='bf16',peak_source='synthetic',useful_ops=1,
                  executed_ops=1,resources=[],assumptions=['no peak'])
        with self.assertRaisesRegex(ValueError,'requires peak_tops'): roofline.calculate(spec)

    def test_useful_ops_cannot_exceed_executed(self):
        spec=dict(engine='cube',dtype='bf16',peak_tops=10,peak_source='synthetic',
                  useful_ops=2,executed_ops=1,resources=[],assumptions=['impossible'])
        with self.assertRaisesRegex(ValueError,'exceeds executed_ops'): roofline.calculate(spec)


    def test_documented_causal_formula_matches_gemm_counting(self):
        small = mfu.gemm_ops(dict(m=8, n=8, k=64, mask='causal'))
        self.assertEqual(small['useful_ops'], 64 * 8 * 9)   # d*S(S+1)，不是 S^2*d
        big = mfu.gemm_ops(dict(m=1024, n=1024, k=64, mask='causal'))
        self.assertEqual(big['useful_ops'], 64 * 1024 * 1025)
        pairs = 1024 * 1025 // 2
        # 归约维掩码套输出掩码公式 2*K*N_valid 会高估 S/d 倍
        self.assertEqual(2 * 1024 * pairs // big['useful_ops'], 1024 // 64)


if __name__ == '__main__':
    unittest.main()
