"""Drift checks between the Markdown hardware/peak tables and their JSON data."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / 'assets/skills/ascend-roofline'


def cells(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def markdown_rows(path, first_cell):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('|'):
            row = cells(line)
            if row and first_cell.match(row[0]):
                rows.append(row)
    return rows


class HardwareTableTest(unittest.TestCase):
    def setUp(self):
        self.hardware = json.loads((SKILL / 'assets/hardware.json').read_text(encoding='utf-8'))
        self.rows = markdown_rows(SKILL / 'references/hardware.md', re.compile(r'^950[A-Za-z0-9_]+$'))

    def test_every_variant_matches_its_markdown_row(self):
        self.assertEqual(len(self.rows), len(self.hardware['variants']))
        table = {row[0]: row for row in self.rows}
        for variant in self.hardware['variants']:
            name = variant['config_file'].removeprefix('Ascend').removesuffix('.ini')
            self.assertIn(name, table)
            raw = variant['raw_config']
            soc = raw['SoCInfo']
            expected = [
                name,
                f"{soc['cube_core_cnt']}/{soc['vector_core_cnt']}",
                f"{round(int(soc['memory_size']) / 2**30):g}",
                f"{round(int(soc['l2_size']) / 2**20):g}",
                f"{raw['AICoreSpec']['cube_freq']}/{raw['VectorCoreSpec']['vec_freq']}",
                raw['AICoreMemoryRates']['ddr_rate'],
                raw['VectorCoreMemoryRates']['ddr_rate'],
            ]
            self.assertEqual(table[name], expected, name)

    def test_shared_capacity_and_unverified_peaks_stay_uniform(self):
        seen = set()
        for variant in self.hardware['variants']:
            core = variant['raw_config']['AICoreSpec']
            vector = variant['raw_config']['VectorCoreSpec']
            seen.add((core['l1_size'], core['l0_a_size'], core['l0_b_size'], core['l0_c_size'],
                      vector['ub_size'], vector['ubblock_size'], vector['ubbank_size'],
                      vector['ubbank_num'], vector['ubsubbank_num'], vector['vec_calc_size'],
                      vector['vector_reg_width']))
        self.assertEqual(len(seen), 1, 'capacity constants drifted between variants')
        self.assertTrue(all(v['verified_l2_bandwidth_gbps'] is None for v in self.hardware['variants']))


class PeakTableTest(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((SKILL / 'assets/peaks.json').read_text(encoding='utf-8'))
        rows = markdown_rows(SKILL / 'references/peaks.md', re.compile(r'^950(PR|DT)$'))
        # Cube rows have seven cells; the Vector table adds INT64.
        self.cube = {(row[0].lower(), row[1].split('/')[0]): row for row in rows if len(row) == 7}
        self.vector = {(row[0].lower(), row[1].split('/')[0]): row for row in rows if len(row) == 8}

    def test_cube_table_matches_catalog(self):
        expected = 0
        for device, variants in self.catalog['devices'].items():
            for aic, entry in variants.items():
                expected += 1
                row = self.cube[(device, aic)]
                self.assertEqual(row[1], f"{entry['aic_cores']}/{entry['aiv_cores']}", device)
                peak = entry['cube']
                self.assertEqual(float(row[2]), peak['fp16'], (device, aic))
                self.assertEqual(float(row[3]), peak['tf32'], (device, aic))
                self.assertEqual(float(row[4]), peak['fp8'], (device, aic))
                self.assertEqual(float(row[5]), peak['mxfp4'], (device, aic))
                self.assertEqual(float(row[6]), peak['int8'], (device, aic))
                self.assertEqual(peak['bf16'], peak['fp16'], (device, aic))
                self.assertEqual(peak['mxfp8'], peak['fp8'], (device, aic))
                self.assertEqual(peak['hif8'], peak['fp8'], (device, aic))
        self.assertEqual(len(self.cube), expected)

    def test_vector_table_matches_catalog(self):
        expected = 0
        for device, variants in self.catalog['devices'].items():
            for aic, entry in variants.items():
                expected += 1
                row = self.vector[(device, aic)]
                self.assertEqual(row[1], f"{entry['aic_cores']}/{entry['aiv_cores']}", device)
                peak = entry['vector']
                self.assertEqual(float(row[2]), peak['fp16'], (device, aic))
                self.assertEqual(float(row[3]), peak['fp32'], (device, aic))
                self.assertEqual(float(row[4]), peak['int8'], (device, aic))
                self.assertEqual(float(row[5]), peak['int16'], (device, aic))
                self.assertEqual(float(row[6]), peak['int32'], (device, aic))
                self.assertEqual(float(row[7]), peak['int64'], (device, aic))
                self.assertEqual(peak['bf16'], peak['fp16'], (device, aic))
        self.assertEqual(len(self.vector), expected)


if __name__ == '__main__':
    unittest.main()
