# GLASS-j official released checkpoints — MVTec AD re-evaluation

仅复评作者发布权重，不训练、不使用DTD、不做正常分布判断、不启动trainer/tester。没有fixed640保证，也不能据此声称发布模型未使用测试集选模。BTAD入口、源码、结果和报告保持不变。

## 来源与明确的权重对应

- [官方仓库及README](https://github.com/cqylunlun/GLASS#data-preparation)明确发布GLASS-j的MVTec AD权重。
- [官方Google Drive目录](https://drive.google.com/drive/folders/1Hjlr-CcXwnhWfrWUCJJCooBI_pMP4N1C)。2026-09-28读取公开目录，确认`results/models/backbone_0/mvtec_<category>/`下每类一个checkpoint；文件名及官方文件ID固化在`release_index.json`。
- 已有官方源码commit：`f788d567db552820d744ca2d93e989265aad8c45`，MIT。复用`glass_external.official.modules`和`adapter.model_for`，不调用`load_last`。源码现存BTAD的3行final_only改动不进入此次推理路径。

| 类别 | 发布文件 | 类别 | 发布文件 | 类别 | 发布文件 |
|---|---|---|---|---|---|
| bottle | ckpt_best_543.pth | cable | ckpt_best_72.pth | capsule | ckpt_best_56.pth |
| carpet | ckpt_best_431.pth | grid | ckpt_best_581.pth | hazelnut | ckpt_best_182.pth |
| leather | ckpt_best_119.pth | metal_nut | ckpt_best_5.pth | pill | ckpt_best_233.pth |
| screw | ckpt_best_139.pth | tile | ckpt_best_369.pth | toothbrush | ckpt_best_543.pth |
| transistor | ckpt_best_52.pth | wood | ckpt_best_637.pth | zipper | ckpt_best_149.pth |

不glob任取第一个、不按测试成绩挑文件；额外版本、缺失或多个匹配目录均报错。`filename_epoch_index`仅记录文件名数字；源码保存时使用零基`i_epoch`，不把数字解释为固定640轮。官方trainer存在测试Image AUROC+Pixel AUROC之和选best的路径；以上命名也为ckpt_best，但发布权重具体选择历史没有独立训练日志证明。

本地已下载并检查**仅bottle**官方文件：15,762,703字节，SHA256 `4d4631625ffece4ed8302b3bfca661b5dc8933da4176d54e1be8c2adfce4e516`。包含discriminator和pre_projection，无骨干权重和epoch字段。两个官方完整尺寸模块及BN全部键严格加载通过；其余14类仅核对公开目录，尚未下载检查内容。

`check`/`predict`生成checkpoint_manifest.csv，记录路径、大小、SHA256、state_dict键、可识别epoch及来源。要求投影器和判别器全部键（包括BN num_batches_tracked）齐全后strict加载，不使用strict=False。若附带backbone或backbone_buffers，完整严格加载；未知模块或不匹配配置/形状拒绝继续。

外部骨干使用[torchvision WideResNet50-2 ImageNet V1](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth)，单独传`--backbone-weights`，可完全离线加载。保持官方构造（包括特征尺寸探测）→加载发布判别器/投影器的顺序；发布文件缺少骨干BN时保留该官方构造结果，不伪称恢复了发布训练时骨干缓冲区。

## 固定推理、数据与评价视野

WideResNet50-2，layer2/layer3，1536/1536维，patchsize3，判别器2层/1024隐藏维，pre_proj1，seed0，FP32、eval、无梯度、TF32关闭。直接调用官方`GLASS._predict`，保留其image_score、双线性上采样和Gaussian sigma4。没有额外sigmoid、归一化或平滑。官方model_for构造了上游优化器对象，但本模块从不调用训练、backward或optimizer.step。

复用`mvtec_ad_records`进行清单与mask配对；标准数据根目录含train/good（共享清单函数需其存在，但不读取训练图像进行拟合）和test/ground_truth。全部15类检查测试数1725、正常467、异常1258；不符时报错，不增删样本凑数。支持`--category bottle`等任意单类。

RGB使用官方MVTec TEST `transform_img`，普通类别短边Resize288，toothbrush/wood短边329，之后CenterCrop288。逐图记录原尺寸、实际torchvision缩放尺寸、round-to-even裁剪坐标和输出尺寸，不把所有图假设成方形。未读取真实GT的RGB函数不使用标签进行任何决策。

默认评价身份为 **official_fov_288_unified_metrics**：预测为官方288×288；GT从原图mask非零二值化、按同一resize尺寸最近邻缩放、同一位置裁剪。异常图即使原mask或裁剪后mask为空，仍保留原图像标签。GT在该类全部预测保存后读取。

禁止将GLASS裁剪图拉伸回全图或边界延拓。本实验与旧BTAD或DINO全图H//4、W//4报告的像素视野不同，不能直接相减声称共同视野效果。

## 独立入口及服务器命令

沿用服务器已有`glass_btad`环境；额外只有自动下载需要gdown。下面命令供用户执行，彼此独立，不自动串联、不自动关机。每个阶段在独立输出目录保存run.log、exit_code、config.json、source_snapshot；失败保存failure.json，已保存预测不会删除。命令里的路径是示例，代码不硬编码服务器或G盘路径。

### 1. 下载或手动放置权重

安装可选下载依赖及运行小测试（不会下载权重/数据或正式评测）：

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 &&
/root/miniconda3/envs/glass_btad/bin/python -m pip install gdown==5.2.0 &&
/root/miniconda3/envs/glass_btad/bin/python -m pytest DINOv3/tests/test_glass_mvtec_pretrained.py -q
```

自动下载只读取已确认的15个官方文件ID，写新目录，不清空原GLASS/results：

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
/root/miniconda3/envs/glass_btad/bin/python -u -m external_baselines.glass_mvtec_pretrained.runner download \
  --category all \
  --weights-root /root/autodl-tmp/weights/glass_j_mvtec_release_v1 \
  --output-dir /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_download_v1
```

下载失败时不更换来源。保留已下载部分；手工从上面的官方Drive下载results目录，放到新的weights-root，再直接运行check。支持weights-root指向results的父目录、results本身、models、backbone_0；必须唯一匹配上述发布文件。download不覆盖已有目录，不提供自动删目录“重试”。Google Drive访问权限/配额可能导致自动下载失败。

DTD、BTAD数据、训练前景mask均不需要。ImageNet V1文件可复用上一轮`/root/autodl-tmp/weights/wide_resnet50_2-95faca4d.pth`，不会自动联网补骨干。

### 2. 单类检查

check在CPU构造官方模型并严格加载权重，检查数据和裁剪；不运行完整预测或CUDA指标。先bottle；对特殊预处理可独立改为wood或toothbrush，使用另一输出目录。正式predict会自行检查全部请求类别输入。

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 &&
/root/miniconda3/envs/glass_btad/bin/python -u -m external_baselines.glass_mvtec_pretrained.runner check \
  --category bottle \
  --dataset-root /root/autodl-tmp/datasets/mvtecAD \
  --weights-root /root/autodl-tmp/weights/glass_j_mvtec_release_v1 \
  --backbone-weights /root/autodl-tmp/weights/wide_resnet50_2-95faca4d.pth \
  --output-dir /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_check_bottle_v1
```

### 3. 15类预测

单卡（例如T4），默认batch4；显存不足可降为1，不改变图像几何。每次只保留一类模型，按batch保存float32 NPZ；不把全部数据集原图/预测积累到GPU。可用同一命令的`--category bottle`先独立检查预测，输出目录另命名；不因单类结果调整参数。

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 &&
/root/miniconda3/envs/glass_btad/bin/python -u -m external_baselines.glass_mvtec_pretrained.runner predict \
  --category all --device cuda:0 --batch-size 4 \
  --dataset-root /root/autodl-tmp/datasets/mvtecAD \
  --weights-root /root/autodl-tmp/weights/glass_j_mvtec_release_v1 \
  --backbone-weights /root/autodl-tmp/weights/wide_resnet50_2-95faca4d.pth \
  --output-dir /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_predict_v1
```

### 4. 独立统一评价

只读predict保存的图像分数、浮点图和二值GT；不需要原图、模型权重、DTD，不初始化GLASS。仍需GPU：调用现有`mvtec_broad6_compose2.metrics.evaluate_fast`，禁用CPU fallback；Image AUROC/AP用保存的官方image_score通过`mvtec_task_decoupled.runner.image_metrics`重算覆盖默认map-max代理。

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 &&
/root/miniconda3/envs/glass_btad/bin/python -u -m external_baselines.glass_mvtec_pretrained.runner evaluate \
  --category all --device cuda:0 \
  --prediction-root /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_predict_v1 \
  --output-dir /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_evaluate_v1
```

AP=average precision；AUPRO@0.3用200个类别全局分数范围线性阈值、四连通。固定FPR≤1%/5%复用`normal_reconstruction.scoring.fixed_fpr_diagnostics`，小区域≤当前288图面积0.1%。macro复用现有类别等权及mean_available/N/A口径，单类明确category_count=1、all_15_completed=false。空小区域类别不进该项平均分母。输出固定FPR实际值、召回、区域及小区域覆盖、区域/图片数量。阈值不参与模型选择。

### 5. 可选共同视野比较

直接接收已有MVTec `guided_validation`完整输出及其中BASE/GUIDED/SMOOTH之一，以下路径需要换成服务器实际保留的全量GUIDED结果目录（含预测图，不是轻量报告包）。

```bash
cd /root/autodl-tmp/GLASS-j-Reevaluation &&
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 &&
/root/miniconda3/envs/glass_btad/bin/python -u -m external_baselines.glass_mvtec_pretrained.runner compare \
  --category all --device cuda:0 \
  --prediction-root /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_predict_v1 \
  --guided-reference /root/autodl-tmp/GLASS-j-Reevaluation/DINOv3/outputs/mad_equal/guided_validation_mvtec_full_v1 \
  --reference-method GUIDED \
  --output-dir /root/autodl-tmp/GLASS-j-Reevaluation/outputs/glass_mvtec_release_compare_guided_v1
```

其他全图连续预测可替换`--guided-reference`为`--reference-index /absolute/path/reference.csv`。CSV必须有：

```csv
category,relative_path,map_path,map_key,image_score,coordinate_system
bottle,bottle/test/good/000.png,/absolute/path/prediction.npz,GUIDED,1.234,full_frame
```

map_path支持NPZ指定map_key或单数组NPY（map_key空），可相对CSV目录；image_score必须为该方法原始图像分数。category+relative_path唯一且与GLASS所选测试集完全对应。只接受声明full_frame的浮点图，不接受热力图、已裁剪图或旧汇总指标。参考全图分数先双线性align_corners=False映射到每图resized_hw，再按相同crop坐标裁288；使用GLASS保存的同一份二值GT重算双方五指标及固定FPR。双方图像分数不改变；共同视野差值方向为REFERENCE−GLASS，输出独立表，不覆盖旧报告。

没有参考预测时不执行compare，GLASS自身evaluate仍可独立完成并记录compare_executed=false。

## 保存、复评所需文件与资源

predict保存checkpoint_manifest、backbone身份、配置/依赖/命令、源码快照、逐图分数和geometry、每图float32 anomaly_map、原图image_label与uint8二值GT、原mask和裁剪后空mask审计、data_counts、日志、resources、exit_code及complete。预测完成后读取GT，GT写入失败仍保留预测NPZ和部分索引；失败目录不自动覆盖或清理。

evaluate按类别将该类预测加载到CPU用于统一指标，不同时保留15类像素列表。resources分别记录模型加载、输入处理、官方推理（含官方返回CPU分数图）、落盘、GT准备、评价；记录GPU型号及allocated/reserved峰值。不把纯推理阶段时间当作含IO/GT/评价的端到端延迟。T4真实显存与耗时尚未验证。

- **轻量报告包**：CSV、JSON、MD、日志和源码快照。可审阅指标，不能仅凭此重算像素结果。
- **可重算评价包**：完整predict目录（maps、gt、索引、几何、完成记录）与evaluate/compare结果。
- **可重新推理包**：再加发布15类checkpoint、ImageNet V1骨干、原MVTec数据和依赖环境。权重及预测不自动清理。

## 最小同步与验证边界

本独立仓库已包含所需公共函数，无需原完整工作区。原工作区增量同步时仅需新增`external_baselines/glass_mvtec_pretrained/`和`DINOv3/tests/test_glass_mvtec_pretrained.py`。依赖现有`external_baselines/GLASS`、`glass_external`及项目内数据/CSV/指标模块；不改它们。若服务器未有上一轮代码，需同时同步这些原有依赖。离线check/predict/evaluate不需要gdown，只自动download需要它。

实际本地验证：新增专项12项＋原BTAD回归7项，**19 passed，14.75秒**。覆盖普通类别和toothbrush/wood官方RGB变换、整数裁剪、二值GT与裁后空mask、BN缺键拒绝、明确文件选择、保存重载一致性、官方image_score覆盖map-max、单类macro与小区域N/A、纯预测/离线评价/共同视野编排。根入口及5个子入口的实际模块CLI帮助通过；5个Python文件内存编译、README的6段Bash静态语法检查通过。

其中一项CUDA测试使用**真实官方WideResNet结构、1536维模块、随机骨干、64×64合成输入和单batch**，包装输出与官方`_predict`一致；不代表真实预训练骨干或288实图性能。CPU小模型端到端编排测试的五指标后端使用mock，不能当成正式CUDA指标验证。自动下载入口测试使用mock下载器；另外通过官方Google Drive HTTPS实际取得一个bottle文件并严格检查完整尺寸的两个官方头部。gdown网络下载、其余14类权重内容、ImageNet V1骨干完整恢复、真实MVTec空间对应、T4显存/耗时及15类CUDA指标仍未验证。

尚未运行真实15类预测/评价，未连接服务器、上传、执行Git或修改BTAD源码、旧结果及报告。合成测试不等同于论文数值复现。

## 独立仓库说明

本仓库仅导出本实验及必要公共函数。`glass_external`只保留模型构造、官方模块导入及汇总，不提供BTAD训练入口。原README中的19项测试指开发工作区的联合回归；独立仓库执行本轮专项测试。原服务器示例使用的glass_btad环境可复用，也可自行创建Python3.10环境。仓库不包含上文提到的本地bottle检查权重。
