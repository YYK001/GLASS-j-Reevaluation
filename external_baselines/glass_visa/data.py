"""Reuse raw CSV records and the complete official VisADataset image/synthesis path."""
from pathlib import Path
from external_baselines.glass_external.official import modules
from external_baselines.patchcore_official_eval.data import records, metadata, evaluation_mask
from external_baselines.glass_external.data import geometry, map_to_evaluation

upstream = modules()


class VisADataset(upstream.visa.VisADataset):
    def __init__(self, items, dtd_root, *, training, foreground=None, distribution_mode=2, size=288):
        self.items = tuple(items)
        self.foreground = foreground or {}
        self.foreground_reads = []
        if not self.items or len({r.category for r in self.items}) != 1:
            raise ValueError('One nonempty category required')
        if training and any(r.label != 0 or r.mask_path is not None for r in self.items):
            raise ValueError('Training uses all and only official normal images, without defect masks')
        if distribution_mode not in (2,3):
            raise ValueError('Explicit official Excel branch 2/3 required')
        super().__init__(source='', anomaly_source_path=str(dtd_root), dataset_name='visa',
            classname=self.items[0].category, resize=size, imagesize=size,
            split=upstream.visa.DatasetSplit.TRAIN if training else upstream.visa.DatasetSplit.TEST,
            distribution=0, fg=1, mean=.5, std=.1, rand_aug=1, batch_size=8)
        # Construct with official distribution=0 (short-side Resize); modes 2/3 only bypass
        # trainer's relative XLSX lookup. They select exactly the recorded official branch.
        self.distribution = distribution_mode
        if training and not self.anomaly_source_paths:
            raise FileNotFoundError('DTD images/<texture>/*.jpg required')
        if training and any(str(Path(r.path).resolve()) not in self.foreground for r in self.items):
            raise FileNotFoundError('Incomplete checked normal foreground mask mapping')

    def foreground_mask_path(self, image_path):
        path = self.foreground[str(Path(image_path).resolve())]
        self.foreground_reads.append(dict(image_path=image_path, foreground_path=path))
        return path

    def get_image_data(self):
        groups = {self.classname:{kind:[r.path for r in self.items if r.label == label]
                                 for kind,label in (('normal',0),('anomaly',1))}}
        # Labels/GT are not read by prediction. Evaluation reads the CSV-associated original
        # mask separately, after official scores/maps have been saved.
        return groups, [[self.classname,'anomaly' if r.label else 'normal',r.path,None] for r in self.items]
