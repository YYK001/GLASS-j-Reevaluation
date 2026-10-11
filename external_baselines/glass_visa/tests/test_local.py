import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from external_baselines.glass_visa import protocol
from external_baselines.glass_visa.cli import parser
from external_baselines.glass_visa.runner import new_output, runtime


class LocalTests(unittest.TestCase):
    def test_actual_official_workbook_and_fixed_foreground(self):
        table = protocol.branches()
        expected = dict(zip(protocol.CATEGORIES,(0,0,1,0,1,0,0,1,1,0,1,1)))
        self.assertEqual({k:r['excel_distribution'] for k,r in table.items()},expected)
        for c,row in table.items():
            self.assertEqual(row['explicit_distribution_mode'],2+expected[c])
            self.assertEqual(row['actual_fg'],1)  # Excel Foreground=0 does not override run-visa fg=1.

    def test_workbook_missing_duplicate_and_nonbinary_fail(self):
        with ZipFile(protocol.EXCEL) as z:
            raw = z.read('xl/worksheets/sheet1.xml')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'invalid.xlsx'
            for corrupt in (raw.replace(b'visa_candle',b'visa_capsules'),
                            raw.replace(b'<v>0</v>',b'<v>2</v>',1)):
                with ZipFile(p,'w') as z: z.writestr('xl/worksheets/sheet1.xml',corrupt)
                with self.assertRaises(ValueError): protocol.branches(p)
            with self.assertRaises(FileNotFoundError): protocol.branches(Path(d)/'missing.xlsx')

    def test_foreground_pairing_is_explicit_and_never_defect_GT(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'fg/candle').mkdir(parents=True)
            items = [SimpleNamespace(category='candle',path=str(root/f'{i:03d}.JPG'),label=0,mask_path=None) for i in range(3)]
            (root/'fg/candle/000.png').touch(); (root/'fg/candle/001.JPG').touch()
            mapping,rows,issues = protocol.foreground_inventory(items,root/'fg')
            self.assertEqual(len(mapping),2); self.assertEqual(issues[0]['status'],'missing')
            self.assertEqual(rows[0]['matching'],'unique_stem')
            (root/'fg/candle/000.bmp').touch()
            self.assertEqual(protocol.foreground_inventory(items,root/'fg')[2][0]['status'],'ambiguous')
            items[0].mask_path = 'real_defect_gt.png'
            with self.assertRaises(ValueError): protocol.foreground_inventory(items,root/'fg')

    def test_shared_defaults_and_upstream_only_path_hook(self):
        adapter = ast.parse((protocol.ROOT.parent/'glass_external/adapter.py').read_text())
        funcs = {n.name:n for n in adapter.body if isinstance(n,ast.FunctionDef)}
        defaults = dict(zip([arg.arg for arg in funcs['train_last'].args.kwonlyargs],
                            [ast.unparse(x) if x else None for x in funcs['train_last'].args.kw_defaults]))
        self.assertEqual(defaults['dataset_prefix'],"'btad'")
        self.assertEqual(defaults['protocol_id'],'PROTOCOL')
        visa = ast.parse((protocol.ROOT/'datasets/visa.py').read_text())
        cls = next(n for n in visa.body if isinstance(n,ast.ClassDef) and n.name=='VisADataset')
        names = {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertIn('self.foreground_mask_path(image_path)',ast.unparse(names['__getitem__']))
        self.assertIn("'fg_mask/'",ast.unparse(names['foreground_mask_path']))
        self.assertEqual(protocol.SOURCE['commit'],'f788d567db552820d744ca2d93e989265aad8c45')

    def test_cli_help_imports_without_scientific_environment(self):
        result = subprocess.run([sys.executable,'-m','external_baselines.glass_visa','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('summarize',result.stdout)
        args = parser().parse_args(['evaluate','--output-dir','example','--prediction-root','saved'])
        self.assertEqual(args.inference_batch,1)
        self.assertEqual(protocol.selected(['all']),protocol.CATEGORIES)
        with self.assertRaises(ValueError): protocol.selected(['candle','candle'])

    def test_new_outputs_do_not_overwrite_inputs_or_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = root/'data'; data.mkdir()
            args = SimpleNamespace(output_dir=root/'out',dataset_root=data,stage='check',categories=['candle'])
            new_output(args)
            with self.assertRaises(FileExistsError): new_output(args)
            args.output_dir = data/'output'
            with self.assertRaises(ValueError): new_output(args)
            args.output_dir = root
            with self.assertRaises(ValueError): new_output(args)
        self.assertEqual(sum(c[0] for c in protocol.COUNTS.values()),8659)
        self.assertEqual(sum(c[1] for c in protocol.COUNTS.values()),962)
        self.assertEqual(sum(c[2] for c in protocol.COUNTS.values()),1200)


if __name__ == '__main__': unittest.main()
