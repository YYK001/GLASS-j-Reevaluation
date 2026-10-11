"""Small standard-library checks for extracting only official foreground resources."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('glass_visa_setup',Path(__file__).with_name('kaggle_glass_visa.py'))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ForegroundArchiveTests(unittest.TestCase):
    def test_only_visa_extracted_and_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive = root/'foreground.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for c in setup.CATEGORIES: z.writestr('release/fg_mask/'+c+'/0000.png',b'unchanged mask bytes')
                z.writestr('release/fg_mask/bottle/0000.png',b'other dataset')
                z.writestr('release/fg_mask/candle/.directory',b'metadata')
            target = setup.extract_foreground(archive,root/'extracted')
            self.assertEqual(len(list(target.glob('*/*.png'))),12)
            self.assertEqual((target/'candle/0000.png').read_bytes(),b'unchanged mask bytes')
            self.assertFalse((target/'bottle').exists())
            self.assertTrue((target/'archive_paths.json').is_file())

    def test_ambiguous_or_escaping_member_rejected(self):
        for names in (('release/../candle/0000.png',),
                      ('release/candle/0000.png','another/candle/0000.png')):
            with self.subTest(names=names),tempfile.TemporaryDirectory() as d:
                root = Path(d); archive = root/'foreground.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for name in names: z.writestr(name,b'mask')
                with self.assertRaises(ValueError): setup.extract_foreground(archive,root/'extracted')


if __name__=='__main__': unittest.main()
