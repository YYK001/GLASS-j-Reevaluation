"""Small real official CPU training on synthetic data when dependencies exist.

These checks do NOT validate pretrained WideResNet, CUDA resources or formal metrics.
The existing GLASS regression suites remain the source for generic GLASS checks.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in
                ('torch','torchvision','numpy','PIL','pandas','imgaug','timm','tensorboard','pytest','cv2'))


@unittest.skipUnless(AVAILABLE,'GLASS scientific environment unavailable; runtime tests not executed')
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        import csv
        import numpy as np
        import torch
        from PIL import Image
        from external_baselines.glass_visa.protocol import CATEGORIES
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.visa,self.fg,self.dtd = self.root/'visa',self.root/'fg',self.root/'dtd'
        rows = []; rng = np.random.default_rng(7)
        for c in CATEGORIES:
            for split,label,count in (('train','normal',10),('test','anomaly',2),('test','normal',1)):
                for i in range(count):
                    name = f'{c}/Data/Images/{split}_{label}/{i:03d}.JPG'
                    image = self.visa/name; image.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rng.integers(0,256,(64,96,3),dtype=np.uint8)).save(image)
                    mask = ''
                    if label=='anomaly':
                        mask = f'{c}/Data/Masks/{i:03d}.png'
                        path = self.visa/mask; path.parent.mkdir(parents=True,exist_ok=True)
                        array = np.zeros((64,96),np.uint8); array[20:28,40:48] = 2+i
                        Image.fromarray(array).save(path)
                    elif split=='train':
                        path = self.fg/c/f'{i:03d}.png'; path.parent.mkdir(parents=True,exist_ok=True)
                        array = np.zeros((64,96),np.uint8); array[8:56,24:72] = 255
                        Image.fromarray(array).save(path)
                    rows.append(dict(object=c,split=split,label=label,image=name,mask=mask))
        path = self.visa/'split_csv/1cls.csv'; path.parent.mkdir(parents=True)
        with path.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['object','split','label','image','mask']); writer.writeheader()
            # Deliberately interleave images and mask IDs; CSV record association is authoritative.
            writer.writerows(rows[::-1])
        (self.dtd/'noise').mkdir(parents=True)
        Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(self.dtd/'noise/0.jpg')
        from external_baselines.glass_visa.data import records
        _,self.train,self.test,_ = records('visa',self.visa,['candle','cashew'])

    def test_csv_pairing_full_train_official_preprocessing_and_foreground_read(self):
        import numpy as np
        import torch
        from unittest.mock import patch
        from external_baselines.glass_external.official import modules
        from external_baselines.glass_visa.data import VisADataset, evaluation_mask, metadata
        from external_baselines.glass_visa.protocol import foreground_inventory
        self.assertEqual(len(self.train['candle']),10)  # Including index 0 and 8.
        mapping,_,issues = foreground_inventory(self.train['candle'],self.fg); self.assertFalse(issues)
        ds = VisADataset(self.train['candle'],self.dtd,training=True,foreground=mapping,size=64)
        self.assertIs(VisADataset.__getitem__,modules().visa.VisADataset.__getitem__)
        self.assertIs(VisADataset.rand_augmenter,modules().visa.VisADataset.rand_augmenter)
        with patch.object(modules().visa,'perlin_mask',wraps=modules().visa.perlin_mask) as perlin:
            item = ds[0]
        self.assertEqual(ds.class_fg,1); self.assertEqual(len(ds.foreground_reads),1)
        self.assertTrue(torch.isfinite(item['aug']).all())
        self.assertTrue((perlin.call_args.args[4] == 0).any())
        self.assertTrue((perlin.call_args.args[4] == 1).any())
        official = modules().visa.VisADataset(str(self.visa),str(self.dtd),classname='candle',resize=64,imagesize=64,fg=1)
        from PIL import Image
        image = Image.open(self.train['candle'][0].path).convert('RGB')
        torch.testing.assert_close(ds.transform_img(image),official.transform_img(image),rtol=0,atol=0)
        for r in self.test['candle']:
            if r.label:
                self.assertEqual(Path(r.path).stem,Path(r.mask_path).stem)
                row = metadata(r,self.visa,'test')
                self.assertTrue(evaluation_mask(row,self.visa).any())  # Defect IDs 2/3 are foreground.

    def test_real_official_update_excel_branches_and_strict_last_reload(self):
        import numpy as np
        import torch
        from unittest.mock import patch
        from DINOv3.tests.test_glass_btad_external import small_model
        from external_baselines.glass_external import adapter
        from external_baselines.glass_external.official import modules
        from external_baselines.glass_visa.data import VisADataset
        from external_baselines.glass_visa.protocol import branches,foreground_inventory,PROTOCOL
        for c in ('candle','cashew'):
            branch = branches()[c]; mapping,_,_ = foreground_inventory(self.train[c],self.fg)
            ds = VisADataset(self.train[c][:2],self.dtd,training=True,foreground=mapping,
                             distribution_mode=branch['explicit_distribution_mode'],size=64)
            dl = torch.utils.data.DataLoader(ds,batch_size=2,shuffle=False)
            model = small_model(); target = self.root/c; target.mkdir()
            model.set_model_dir(str(target/'official'),'visa_'+c)
            before = {k:v.clone() for k,v in model.pre_projection.state_dict().items()}
            fail = AssertionError('No spectrum rejudgement, relative Excel fallback, periodic prediction or test selection')
            with patch.object(model,'predict',side_effect=fail), patch.object(model,'_evaluate',side_effect=fail), \
                 patch.object(modules().glass.pd,'read_excel',side_effect=fail), \
                 patch.object(modules().utils,'distribution_judge',side_effect=fail):
                checkpoint = adapter.train_last(model,dl,c,target,formal=False,protocol_id=PROTOCOL,dataset_prefix='visa')
            self.assertEqual(model.svd,branch['excel_distribution'])
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.pre_projection.state_dict().items()))
            images = torch.stack([ds[i]['image'] for i in range(2)])
            scores,maps = model._predict(images)
            reloaded = small_model()
            adapter.load_last(reloaded,checkpoint,c,require_formal=False,protocol_id=PROTOCOL)
            scores2,maps2 = reloaded._predict(images)
            np.testing.assert_array_equal(scores,scores2); np.testing.assert_array_equal(maps,maps2)
            with self.assertRaises(ValueError): adapter.load_last(reloaded,checkpoint,c,protocol_id=PROTOCOL)
            with self.assertRaises(ValueError): adapter.load_last(reloaded,checkpoint,c,require_formal=False)
            model.logger.logger.close()

    def test_non_square_geometry_cache_and_official_score_preserved(self):
        import numpy as np
        from external_baselines.glass_visa.data import geometry,map_to_evaluation
        from external_baselines.glass_visa.runner import saved_inputs
        from external_baselines.glass_visa.protocol import PROTOCOL
        from external_baselines.patchcore_official_eval.storage import json_write,npz_write
        g = geometry(self.train['candle'][0].path)
        self.assertEqual((g['resized_height'],g['resized_width'],g['crop_top'],g['crop_left']),(288,432,0,72))
        native = np.broadcast_to(np.linspace(0,1,288,dtype=np.float32),(288,288)).copy()
        mapped = map_to_evaluation(native,g)
        self.assertEqual(mapped.shape,(16,24)); self.assertEqual(float(mapped[0,0]),0); self.assertEqual(float(mapped[0,-1]),1)
        path = self.root/'cached'; path.mkdir()
        row = dict(category='candle',relative_path='candle/image.JPG',label=1,image_score=0.123456789,
                   prediction_file='maps/000000.npz',evaluation_hw=[16,24])
        json_write(path/'samples.json',[row]); json_write(path/'complete.json',
            dict(protocol_id=PROTOCOL,status='complete',formal_model=True,image_count=1))
        npz_write(path/row['prediction_file'],input_map=native,evaluation_map=mapped,mask=mapped>.5,
            relative_path=np.asarray(row['relative_path']), category=np.asarray('candle'),label=np.asarray(1),
            image_score=np.asarray(row['image_score'],np.float64))
        rows,maps,masks = saved_inputs(path,'candle')
        self.assertEqual(rows[0]['image_score'],row['image_score'])
        self.assertNotEqual(rows[0]['image_score'],float(maps[0].max()))
        np.testing.assert_array_equal(maps[0],mapped); np.testing.assert_array_equal(masks[0],mapped>.5)

    def test_existing_macro_NA_and_disjoint_batch_merge(self):
        from external_baselines.glass_visa.runner import write_summary
        from external_baselines.patchcore_official_eval.storage import read_json
        import csv
        rows = [dict(category=c,**{k:v for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO_at_0p3')})
                for c,v in (('candle',.2),('cashew',.8))]
        fixed = [dict(category=c,fpr_cap=cap,actual_fpr=cap,defect_pixel_recall=.5,region_mean_coverage=.4,
            small_region_mean_coverage=None if c=='candle' else .6,region_count=2,small_region_count=0 if c=='candle' else 1,
            small_region_image_count=0 if c=='candle' else 1) for cap in (.01,.05) for c in ('candle','cashew')]
        out = self.root/'summary'; out.mkdir(); write_summary(out,rows,fixed,('candle','cashew'))
        with (out/'macro_metrics.csv').open() as f: macro = list(csv.DictReader(f))[0]
        self.assertEqual(float(macro['image_AP']),.5); self.assertIn('NOT_full_dataset',macro['scope'])
        with (out/'fixed_fpr_macro.csv').open() as f: fm = list(csv.DictReader(f))[0]
        self.assertEqual(float(fm['small_region_mean_coverage']),.6)
        self.assertEqual(int(fm['small_region_mean_coverage_valid_category_count']),1)
        with self.assertRaises(ValueError): write_summary(out,rows+rows[:1],fixed,('candle','cashew'))


if __name__ == '__main__': unittest.main()
