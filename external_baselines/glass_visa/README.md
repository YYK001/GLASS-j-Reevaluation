# GLASS-j 官方实现的 VisA 固定末轮复评

独立入口：`python -m external_baselines.glass_visa`。只支持 VisA 官方原始格式的 `split_csv/1cls.csv`，12 类各自训练。训练使用完整正常训练集，DTD 只参与官方纹理合成。不调用 DINOv3、MAD、LOCAL 或 GUIDED 模型；工程中的这些目录只提供已有数据、指标和汇总函数。

本次代码接入不代表正式实验完成。没有连接服务器、下载大型权重、启动正式训练、推送 Git 或改动已有报告和结果。

## 来源与固定定义

复用现有 [GLASS 官方仓库](https://github.com/cqylunlun/GLASS)，commit **f788d567db552820d744ca2d93e989265aad8c45**；MIT 许可证保留在 `../GLASS/LICENSE`，来源及本地兼容改动记录在 `../glass_external/source.json`。这是归档 checkout，不假定存在 `.git`。

以 `../GLASS/shell/run-visa.sh` 和当前源码为准：每类 seed=0，ImageNet **V1** WideResNet50-2，layer2/layer3，pretrain/target embedding=1536，patchsize=3，短边 Resize=288、CenterCrop=288，batch=8，meta_epochs=640，dsc_layers=2、hidden=1024，pre_proj=1，mining=1，noise=0.015，radius=0.75，p=0.5，step=20，limit=392，mean=0.5、std=0.1、rand_aug=1、**fg=1、distribution=0**。其余沿用官方默认，包括 lr=1e-4、margin=0.5、downsampling=8、零额外图像增强参数。FP32，不使用 AMP。实际模型和数据默认参数由源码签名提取，写入配置。

每个 meta epoch 保留官方行为：中心计算遍历完整 loader；判别器更新后检查 `sample_num > 392`，达到后停止本轮判别器迭代。不是把 640 轮改写成 640 次完整梯度遍历。

采用已有 `final_only=True` 接入：不周期读取测试集，不评价或选择 best checkpoint，固定保存第 640 轮 `ckpt_last.pth`。包含判别器、投影器、冻结 backbone 的相关 buffers、类别、源码 commit、完成轮数及实际分支；冻结 backbone 权重另引用同一个 V1 文件，文件身份在配置中核对。正式评价显式加载该文件，严格加载参数与 buffers，拒绝 smoke、错误类别和 BTAD 权重协议。无需调用官方 `main.py --test ckpt` 或搜索 `ckpt_best`。

相对原论文协议的差异必须保留在报告中：**固定末轮替代测试集 best checkpoint 选择；全图 H//4×W//4 复评及裁剪外延拓；项目快速 CUDA 指标**。不能把本入口称为论文最佳权重协议或官方指标逐位复现。

## 数据与前景资源

参考数量如下；不匹配时保存实际路径清单和数量后停止，不补删图片，不使用 index%8 calibration 划分。

| 类别 | 正常训练 | 正常测试 | 异常测试 | Excel Distribution | 实际官方训练模式 |
|---|---:|---:|---:|---:|---|
| candle | 900 | 100 | 100 | 0 | 2，manifold |
| capsules | 542 | 60 | 100 | 0 | 2，manifold |
| cashew | 450 | 50 | 100 | 1 | 3，hypersphere |
| chewinggum | 453 | 50 | 100 | 0 | 2，manifold |
| fryum | 450 | 50 | 100 | 1 | 3，hypersphere |
| macaroni1 | 900 | 100 | 100 | 0 | 2，manifold |
| macaroni2 | 900 | 100 | 100 | 0 | 2，manifold |
| pcb1 | 904 | 100 | 100 | 1 | 3，hypersphere |
| pcb2 | 901 | 100 | 100 | 1 | 3，hypersphere |
| pcb3 | 905 | 101 | 100 | 0 | 2，manifold |
| pcb4 | 904 | 101 | 100 | 1 | 3，hypersphere |
| pipe_fryum | 450 | 50 | 100 | 1 | 3，hypersphere |
| 总计 | **8659** | **962** | **1200** | | |

CSV 图片和缺陷 mask 从同一记录解析；不依赖独立列表的顺序。VisA 多缺陷 ID 使用已有 GT adapter 的“非零为前景”语义。真实缺陷 GT 不进入训练，推理数据类不读 GT。

训练必须另有**正常物体前景 mask**。使用官方 README 的 [foreground masks 公开下载](https://drive.google.com/file/d/1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8/view?usp=sharing)，在服务器由用户下载并解压，定位包含 VisA 12 个类别目录的那一层。不要用真实异常 GT，也不要改为 fg=0 或临时换分割算法。本轮没有下载或核对实际压缩包目录。

`--foreground-root /path/to/fg_mask` 应指向下述目录；省略时兼容 `<VisA根目录>/fg_mask`：

```text
fg_mask/
  candle/0000.JPG   # 与该类别正常训练图片同名
  capsules/0000.JPG
  ...
```

优先精确文件名，找不到时接受唯一同 stem 的 PNG/JPG/BMP/TIFF；大小写不同亦按唯一 stem 匹配。缺失或歧义均列入 `foreground_issues.json` 并停止。要求前景 mask 与原图尺寸一致、编码为 1/L/RGB，实际使用官方第一通道、Resize、CenterCrop、ToTensor、ceil 路径，不额外二值化。检查裁剪后有前景，避免官方 Perlin 重试陷入空前景循环。每张正常训练图的实际前景路径保存在清单；workers>0 时读取发生在 worker 中，主进程读取观测不能作为全局次数。

读取**绝对路径**的 `../GLASS/datasets/excel/visa_distribution.xlsx`，验证 12 类与 0/1 值，并记录表格值。为避免官方 trainer 对相对路径异常的静默频谱回退，先以 distribution=0 构建官方 VisADataset，再映射 Excel 0→模式2、1→模式3。官方 trainer 的这两个模式与 Excel 分支语义相同，不进行 BTAD 风格频谱判断。Excel 的 Foreground 列仅记录；`run-visa.sh` 的 fg=1 对所有类别生效。

## 推理、坐标与复评

直接调用官方 `GLASS._predict`：保留原始 patch 图像分数、双线性异常图插值和 SciPy Gaussian sigma=4。保存官方原生 **288×288 FP32** 连续异常图。图像分数不从映射后的异常图重新计算。

复用 `glass_external.data.geometry/map_to_evaluation`：记录实际短边 Resize 尺寸和 CenterCrop 偏移，把中心裁剪图放回 resized canvas，裁剪外采用已有**最近裁剪边界值延拓**，再 bilinear、align_corners=False 映射到原图 H//4×W//4。不能把裁剪图直接拉伸为完整原图。裁剪外分数来自延拓，属于模型未直接观察的区域。GT 使用原始 CSV 配对 mask 的已有 nearest 全图规则。

复用 `patchcore_official_eval.evaluation.calculate/summarize`：Image AUROC、Image AP、Pixel AUROC、Pixel AP、AUPRO@0.3；AP 是 average precision；AUPRO 为既有快速 CUDA、200 个全局线性阈值、FPR上限0.3、四连通。没有 CPU 回退。1%/5% FPR 诊断复用 `fixed_fpr_diagnostics`，小区域面积≤评价图面积0.1%，无小区域记 N/A，macro 对有效类别等权平均并记录有效分母。只在12类齐全时输出 `dataset_macro`，子集标记 `selected_categories_macro_NOT_full_dataset`；重复或缺失类别汇总失败。

## 环境与资源准备（用户在服务器执行）

以下都是待执行命令；先将代码放入 `REPO`。独立 Python3.10 环境沿用现有 GLASS 的 torch2.1.2 / torchvision0.16.2 / NumPy1.26.3 兼容配置，避免修改已有环境。没有加入 ONNX 导出等推理无关依赖。环境安装与 T4 兼容性仍须真实验证。

```bash
export REPO=/path/to/experiment
export ENV_DIR=/kaggle/working/glass-visa-venv
export PY="$ENV_DIR/bin/python"
export VISA_ROOT=/path/to/VisA
export DTD_ROOT=/path/to/dtd/images
export FG_ROOT=/path/to/official/fg_mask
export ASSETS=/kaggle/working/glass_visa_assets
export RESULTS=/kaggle/working/glass_visa_runs
export STAMP=$(date -u +%Y%m%dT%H%M%S)_$$
cd "$REPO"
uv python install 3.10
uv venv --python 3.10 --seed "$ENV_DIR"
uv pip install --python "$PY" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python "$PY" -r external_baselines/glass_visa/requirements.txt
uv pip install --python "$PY" pytest
"$PY" -m pip check
mkdir -p "$ASSETS" "$RESULTS/logs"
export WEIGHT="$ASSETS/wide_resnet50_2-95faca4d.pth"
# 若已有这个官方V1文件，直接让 WEIGHT 指向它，跳过下载。
"$PY" -c 'import os,torch; torch.hub.download_url_to_file("https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth", os.environ["WEIGHT"], hash_prefix="95faca4d")'
# DTD沿用已有完整的 images/<texture>/*.jpg，不计入VisA训练样本。
# 官方foreground公开包由用户从上面的官方链接下载、解压；FG_ROOT指向类别目录所在层。
```

若 uv 创建的环境不含 pip，可先 `uv pip install --python "$PY" pip` 再执行 `pip check`。已有兼容环境可直接设置 PY，无需重复安装。权重检查只对这一份 V1 backbone 文件进行身份核对，不构造大型文件哈希闭包。

后续命令使用同一个 Bash 会话中的上述变量和下面函数。日志写入独立目录，打印被执行入口的退出码；非零即停止该步骤。输出目录必须全新，不能覆盖旧结果。

```bash
set -o pipefail
run_logged() {
  local stage_log="$1"
  shift
  "$@" 2>&1 | tee "$stage_log"
  local stage_code=${PIPESTATUS[0]}
  echo "入口退出码：$stage_code"
  return "$stage_code"
}
```

## check 与单类 smoke

`check` 不训练，可以在 CPU 执行；检查完整 CSV 路径、数量、DTD 可解码、foreground 对应关系和裁剪后非空、Excel、依赖导入及 V1 权重严格加载（只初始化 CPU backbone）。不会因为缺前景继续训练。

```bash
run_logged "$RESULTS/logs/check_$STAMP.log" "$PY" -u -m external_baselines.glass_visa check \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" --foreground-root "$FG_ROOT" \
  --backbone-weights "$WEIGHT" --categories all --output-dir "$RESULTS/check_$STAMP" || exit $?

# 真实官方网络；8张正常训练图，1个meta epoch，判别器至少1次更新。
# 只有smoke减少epoch/样本上限，其他正式配置与fg、Excel分支保持不变。
run_logged "$RESULTS/logs/smoke_$STAMP.log" "$PY" -u -m external_baselines.glass_visa smoke \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" --foreground-root "$FG_ROOT" \
  --backbone-weights "$WEIGHT" --categories candle --device cuda:0 --workers 0 \
  --smoke-images 8 --output-dir "$RESULTS/smoke_$STAMP" || exit $?
```

smoke 检查真实投影参数改变、有限损失、末轮保存、严格重载后的官方分数与异常图一致、FP32 保存重读一致，并记录训练/推理峰值显存。只在正常训练图片上进行 smoke 推理，不按测试指标选择配置。smoke 权重有独立标记，正式评价拒绝使用。单卡 OOM 直接报错，不能静默降低 batch、分辨率、轮数或切 AMP。

## 单类正式训练和独立评价

```bash
export TRAIN_ONE="$RESULTS/train_candle_$STAMP"
export EVAL_ONE="$RESULTS/eval_candle_$STAMP"
run_logged "$RESULTS/logs/train_candle_$STAMP.log" "$PY" -u -m external_baselines.glass_visa train \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" --foreground-root "$FG_ROOT" \
  --backbone-weights "$WEIGHT" --categories candle --device cuda:0 --workers 2 \
  --output-dir "$TRAIN_ONE" || exit $?
run_logged "$RESULTS/logs/eval_candle_$STAMP.log" "$PY" -u -m external_baselines.glass_visa evaluate \
  --trained-root "$TRAIN_ONE" --dataset-root "$VISA_ROOT" --backbone-weights "$WEIGHT" \
  --categories candle --device cuda:0 --workers 2 --inference-batch 1 \
  --output-dir "$EVAL_ONE" || exit $?
```

训练只需正常图和对应前景，但会检查全套 CSV 数据路径。没有实现跨会话 optimizer 恢复；中断记录为失败，需新目录重跑未完成类别，不能把未满640轮的权重当正式末轮。已经完成的类别可单独保留并独立评价。

## 双 T4 的 12 类训练、评价与汇总

每张约16GB T4一个独立进程，完整模型在一张卡上，不使用 DDP。不假定显存可相加。以下是互不重叠的6+6示例，**不是已经验证过的耗时均衡或12小时调度**；首次真实 smoke 和资源记录决定是否需要调整类别分配。

```bash
export TRAIN0="$RESULTS/train_gpu0_$STAMP"
export TRAIN1="$RESULTS/train_gpu1_$STAMP"
export EVAL0="$RESULTS/eval_gpu0_$STAMP"
export EVAL1="$RESULTS/eval_gpu1_$STAMP"
GROUP0=(candle cashew fryum macaroni2 pcb2 pcb4)
GROUP1=(capsules chewinggum macaroni1 pcb1 pcb3 pipe_fryum)

run_logged "$RESULTS/logs/train_gpu0_$STAMP.log" "$PY" -u -m external_baselines.glass_visa train \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" --foreground-root "$FG_ROOT" \
  --backbone-weights "$WEIGHT" --categories "${GROUP0[@]}" --device cuda:0 --workers 2 --output-dir "$TRAIN0" &
PID0=$!
run_logged "$RESULTS/logs/train_gpu1_$STAMP.log" "$PY" -u -m external_baselines.glass_visa train \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" --foreground-root "$FG_ROOT" \
  --backbone-weights "$WEIGHT" --categories "${GROUP1[@]}" --device cuda:1 --workers 2 --output-dir "$TRAIN1" &
PID1=$!
wait "$PID0"; CODE0=$?
wait "$PID1"; CODE1=$?
echo "训练退出码：$CODE0 $CODE1"
[ "$CODE0" -eq 0 ] && [ "$CODE1" -eq 0 ] || exit 1

run_logged "$RESULTS/logs/eval_gpu0_$STAMP.log" "$PY" -u -m external_baselines.glass_visa evaluate \
  --trained-root "$TRAIN0" --dataset-root "$VISA_ROOT" --backbone-weights "$WEIGHT" \
  --categories "${GROUP0[@]}" --device cuda:0 --workers 2 --inference-batch 1 --output-dir "$EVAL0" &
PID0=$!
run_logged "$RESULTS/logs/eval_gpu1_$STAMP.log" "$PY" -u -m external_baselines.glass_visa evaluate \
  --trained-root "$TRAIN1" --dataset-root "$VISA_ROOT" --backbone-weights "$WEIGHT" \
  --categories "${GROUP1[@]}" --device cuda:1 --workers 2 --inference-batch 1 --output-dir "$EVAL1" &
PID1=$!
wait "$PID0"; CODE0=$?
wait "$PID1"; CODE1=$?
echo "评价退出码：$CODE0 $CODE1"
[ "$CODE0" -eq 0 ] && [ "$CODE1" -eq 0 ] || exit 1

run_logged "$RESULTS/logs/summary_$STAMP.log" "$PY" -u -m external_baselines.glass_visa summarize \
  --result-roots "$EVAL0" "$EVAL1" --categories all --output-dir "$RESULTS/summary_$STAMP" || exit $?
```

## 缓存复评与输出

```bash
# 无模型前向、无需DTD/foreground/backbone文件。仍使用快速CUDA指标。
run_logged "$RESULTS/logs/reevaluate_$STAMP.log" "$PY" -u -m external_baselines.glass_visa evaluate \
  --prediction-root "$EVAL0" --categories "${GROUP0[@]}" --device cuda:0 \
  --output-dir "$RESULTS/reevaluate_gpu0_$STAMP" || exit $?
```

训练目录保存 `config.json`、来源/依赖环境、完整数据清单与实际数量、`branches.csv`、`foreground_manifest.csv`、资源问题记录；每类保存 `train.log`、`training_scalars.csv`、`ckpt_last.pth`、官方覆盖式 `ckpt.pth`、TensorBoard 日志、分阶段 `resources.csv`、配置和完成状态。仅两个末轮权重副本，不按640轮累积 checkpoint。

评价每类保存：

- `image_scores.csv` / `samples.json`：官方图像分数、CSV图片/GT对应路径、原图尺寸、288输入尺寸、实际 Resize 尺寸、裁剪偏移、H//4×W//4评价尺寸和延拓规则。
- `maps/*.npz`：`input_map`（官方原生FP32）、`evaluation_map`（全图FP32）、`mask`（统一坐标bool GT）、原始官方 `image_score`、类别/路径/标签。原生图为官方已插值与高斯处理的连续图；不是热力图PNG或额外归一化。
- `category_metrics.csv`、`fixed_fpr.csv`、推理/评价资源记录和完成状态。运行根目录保存逐类结果、macro和固定FPR macro；只按类别等权汇总。

可以独立打包文本报告（JSON/CSV/LOG），权重与连续预测另行保留，不自动删除、不自动上传：

```bash
# 先明确需要归档的训练、评价和汇总目录；文本包不含pth、npz、TensorBoard事件。
export TEXT_ZIP="$RESULTS/glass_visa_text_$STAMP.zip"
"$PY" - "$TEXT_ZIP" "$TRAIN0" "$TRAIN1" "$EVAL0" "$EVAL1" "$RESULTS/summary_$STAMP" <<'PY'
from pathlib import Path
import sys, zipfile
dest, *roots = map(Path,sys.argv[1:])
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.log'):
                z.write(p, str(Path(root.name)/p.relative_to(root)))
print(dest)
PY
# 连续预测和权重保留在原目录。若需另行导出，使用不同文件名，确认可用磁盘。
# 例如：tar -czf "$RESULTS/predictions_$STAMP.tar.gz" -C "$RESULTS" "$(basename "$EVAL0")" "$(basename "$EVAL1")"
# 模型同理导出TRAIN0/1，并另外保留同一份V1 backbone权重；本入口不会自动执行。
```

## 本地验证与待验证项

本地 Python3.12 无 torch/torchvision/NumPy/Pillow/pytest 等科学环境；本轮未准备 VisA、官方 foreground 或 V1 权重。执行 Python语法检查、CLI帮助及标准库测试：真实官方 Excel 读取/错误拒绝、类别配置与数量、正常foreground路径配对/缺失/歧义、旧BTAD默认参数和官方路径钩子、输出不覆盖。**6项通过**。

4项运行时专项测试已提供，但本机因缺依赖跳过：CSV与多缺陷ID/真实foreground读取及官方前处理；合成小图上的官方两种分支真实训练更新与严格保存重载；非方形坐标及连续预测/分数缓存一致；现有macro/N/A/重复类别处理。它们不能称为真实WideResNet、T4或正式复现验证。

在兼容环境运行以下命令，包括受共享接入修改影响的两个已有GLASS回归测试集（本机缺pytest/torch，未能执行）：

```bash
"$PY" -m unittest discover -s external_baselines/glass_visa/tests -v
"$PY" -m pytest -q DINOv3/tests/test_glass_btad_external.py DINOv3/tests/test_glass_mvtec_pretrained.py
```

待真实数据和服务器验证：官方前景公开包的实际目录/尺寸与12类覆盖；全部CSV图片/GT对应；V1真实加载和官方WideResNet训练更新；288分辨率batch8在T4上的显存与耗时；真实末轮重载一致性；全12类推理、快速CUDA指标与汇总。本轮不承诺显存或运行时已验证。
