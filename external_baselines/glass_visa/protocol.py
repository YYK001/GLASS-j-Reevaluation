"""Small resource checks; usable without importing a scientific runtime."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1] / 'GLASS'
SOURCE = json.loads((ROOT.parent/'glass_external/source.json').read_text(encoding='utf-8'))
EXCEL = ROOT/'datasets/excel/visa_distribution.xlsx'
PROTOCOL = 'glass_j_visa_final640_seed0_h4_v1'
METHOD = 'GLASS-j 官方实现的 VisA 固定末轮复评'
CATEGORIES = ('candle','capsules','cashew','chewinggum','fryum','macaroni1',
              'macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')
COUNTS = dict(zip(CATEGORIES, ((900,100,100),(542,60,100),(450,50,100),(453,50,100),
    (450,50,100),(900,100,100),(900,100,100),(904,100,100),(901,100,100),
    (905,101,100),(904,101,100),(450,50,100))))
FOREGROUND_URL = 'https://drive.google.com/file/d/1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8/view?usp=sharing'
WEIGHT_URL = 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
WEIGHT_SHA256 = '95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8'


def selected(values):
    if values == ['all']:
        return CATEGORIES
    if not values or len(set(values)) != len(values) or any(v not in CATEGORIES for v in values):
        raise ValueError('Use all or distinct official VisA categories: '+', '.join(CATEGORIES))
    return tuple(values)


def branches(path=EXCEL):
    """Read the fixed upstream workbook explicitly; never swallow XLSX failures."""
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(si.itertext()) for si in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    rows = []
    for row in sheet.findall('.//m:row', ns):
        values = {}
        for cell in row.findall('m:c', ns):
            if cell.find('m:f', ns) is not None:
                raise ValueError('Distribution workbook must contain literal values, not formulas')
            column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
            value = cell.findtext('m:v', default='', namespaces=ns)
            if cell.attrib.get('t') == 'inlineStr':
                value = ''.join(cell.find('m:is', ns).itertext())
            elif cell.attrib.get('t') == 's':
                value = shared[int(value)]
            values[column] = value
        rows.append(values)
    if not rows or tuple(rows[0].get(c) for c in ('A','B','C')) != ('Class','Distribution','Foreground'):
        raise ValueError('Unexpected official VisA workbook headers')
    result = {}
    for row in rows[1:]:
        name = row.get('A', '')
        if not name:
            continue
        c = name.removeprefix('visa_')
        if name != 'visa_'+c or c not in CATEGORIES or c in result:
            raise ValueError('Unknown/duplicate workbook class: '+name)
        if row.get('B') not in ('0','1') or row.get('C') not in ('0','1'):
            raise ValueError('Non-binary official workbook entry: '+name)
        d = int(row['B'])
        result[c] = dict(category=c, requested_distribution=0, excel_distribution=d,
            explicit_distribution_mode=2+d, branch='hypersphere' if d else 'manifold',
            excel_foreground=int(row['C']), actual_fg=1, excel_path=str(Path(path).resolve()))
    if set(result) != set(CATEGORIES):
        raise ValueError('Incomplete VisA distribution workbook')
    return result


def foreground_inventory(items, root):
    """Match normal image to <root>/<category>/<filename>; no defect GT fallback."""
    root = Path(root).resolve()
    mapping, rows, issues = {}, [], []
    used = set()
    candidates = {}
    for r in items:
        image = Path(r.path).resolve()
        if r.label != 0 or r.mask_path is not None:
            raise ValueError('Foreground inventory accepts normal training records only')
        category_dir = root/r.category
        if r.category not in candidates:
            candidates[r.category] = list(category_dir.iterdir()) if category_dir.is_dir() else []
        exact = category_dir/image.name
        hits = [exact] if exact.is_file() else [p for p in candidates[r.category]
            if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
            and p.stem.casefold() == image.stem.casefold()]
        error = '' if len(hits) == 1 else ('missing' if not hits else 'ambiguous')
        if not error and str(hits[0].resolve()) in used:
            error = 'duplicate_foreground_assignment'
        row = dict(category=r.category, image_path=str(image), foreground_path=str(hits[0]) if not error else '',
                   matching='exact_filename' if not error and hits[0] == exact else 'unique_stem', status=error or 'matched')
        rows.append(row)
        if error:
            issues.append(dict(row, candidates=[str(p) for p in hits]))
        else:
            mapping[str(image)] = str(hits[0].resolve())
            used.add(str(hits[0].resolve()))
    return mapping, rows, issues
