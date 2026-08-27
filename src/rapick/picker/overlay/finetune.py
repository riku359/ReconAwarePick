"""Fine-tune CryoTransformer directly from a STAR file (no COCO annotation build step).

This is the fine-tuning stage of Sec. 3.5 and Sec. S2. It is copied from
CryoTransformer's train.py and adapted (MIT, Copyright (c) 2023 Jianlin Cheng,
https://github.com/jianlin-cheng/CryoTransformer, pinned at commit
a56f133f3f499562c32a6bc512eec5f115095b3b). train.py trains from the original COCO-style
train_val_test_data (train/val jpg + annotations/instances_*.json). This script instead
takes any GT-aligned STAR (`_rlnMicrographName/_rlnCoordinateX/_rlnCoordinateY`, the
format of src/rapick/eval/README.md, which the CryoPPP ground_truth/*.star files also
use) plus a directory of micrographs (.mrc or .jpg), and builds targets from it
directly. Because labels are assigned as 0 = particle right here (never read from a
COCO category_id), the collision between category_id=1 and no-object index 1 that
datasets/micrograph.py has to work around cannot recur through this path.

The paper's setting is --finetune_mode head_decoder_encoder_resnet: every weight
trainable, with resnet layer1 always frozen by build_backbone().

*** --resume MUST BE A HEAD-REPAIRED CHECKPOINT -- see the banner in main(). ***
Every weight in the checkpoint is now loaded as-is; nothing is re-initialized. That is
only sound because the repaired class_embed already uses this same 0 = particle scheme.

Optimizer/schedule defaults (--epochs, --lr_drop, --lr, --lr_backbone, --weight_decay,
--clip_max_norm) follow UPicker's fine-tuning stage -- the closest documented reference for
fine-tuning a small transformer-based picker on a handful of labeled micrographs (UPicker,
"Fine-tuning" section / Table 3). --lr/--lr_backbone/
--weight_decay/--clip_max_norm already matched CryoTransformer's own DETR-inherited defaults,
so only --epochs (300 -> 50) and --lr_drop (150 -> 24) changed. The paper text doesn't list
lr/lr_drop, so these were confirmed from the upstream repo
(https://github.com/JachyLikeCoding/UPicker, config/UPICKER/UPICKER_4scale_50epoch.py) via a
temporary clone -- that config is reused unmodified for both UPicker's pretrain and fine-tune
stages (only --lr_backbone is overridden to 0 during pretrain).

Four finetune variants, chosen by --finetune_mode (see configure_finetune_mode):
  head                          -- only class_embed + bbox_embed
  head_decoder                  -- + transformer.decoder + query_embed
  head_decoder_encoder          -- + transformer.encoder + input_proj
  head_decoder_encoder_resnet   -- + backbone (full fine-tune)
"""
import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image

import denoise_micrographs
import util.misc as utils
from datasets.micrograph import make_micrograph_transforms
from engine import train_one_epoch
from models import build_model


def _import_common_2d_metrics():
    """Import calc_common_2d_metrics -- the repository's single STAR reader.

    scripts/setup.sh copies this file over the upstream clone at
    $RAPICK_THIRD_PARTY/cryotransformer/, so a path relative to __file__ resolves only
    while the file still sits in the repository. Try an already-importable module first
    (put <repo>/src/rapick/eval on PYTHONPATH), then the in-repository location.
    """
    try:
        import calc_common_2d_metrics as module
        return module
    except ImportError:
        pass
    # Two layouts to cover. In the repository this file sits at
    # src/rapick/picker/overlay/, so the reader is two levels up in eval/. After
    # setup it sits in the clone at third_party/cryotransformer/, so the reader is
    # up and across. Walking up for the marker covers both without either being
    # spelled out, and covers a clone somewhere else as long as it is under the
    # repository. Beyond that, PYTHONPATH is the answer, and the error says so.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "rapick" / "eval"
        if (candidate / "calc_common_2d_metrics.py").is_file():
            sys.path.insert(0, str(candidate))
            import calc_common_2d_metrics as module
            return module
    sibling = here.parents[2] / "eval"          # the in-repository layout
    if (sibling / "calc_common_2d_metrics.py").is_file():
        sys.path.insert(0, str(sibling))
        import calc_common_2d_metrics as module
        return module
    raise SystemExit(
        "cannot import calc_common_2d_metrics. Put <repo>/src/rapick/eval on "
        "PYTHONPATH before running this script. See src/rapick/picker/README.md.")


_ccm = _import_common_2d_metrics()
load_star_points = _ccm.load_star_points
normalize_mic_name = _ccm.normalize_mic_name


# ---------------------------------------------------------------------------
# STAR -> micrograph dataset
# ---------------------------------------------------------------------------

def match_micrographs_to_star(images_dir: Path, star_path: Path):
    """Pairs every micrograph named in `star_path` with a file under `images_dir`.

    The STAR file is the source of truth for which micrographs train: a micrograph
    present in images_dir but absent from the STAR is excluded rather than treated as
    a hard negative (0 particles), since we cannot distinguish "genuinely no particles"
    from "this file just isn't covered by this STAR" from here. A STAR entry with no
    matching image file is skipped with a warning (e.g. a partial local download).
    """
    points_by_mic = load_star_points(str(star_path))
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in (".mrc", ".jpg", ".jpeg")
    )
    if not image_files:
        raise SystemExit(f"no .mrc/.jpg found under {images_dir}")

    # Raw stem first (matches this repo's own picker output, which never carries a
    # CryoSPARC import hash), normalize_mic_name(name) as fallback (matches GT star
    # naming, which does) -- same two-step resolution as
    # head_repair/cryoppp_gt.py's resolve_train_stem_key.
    stem_to_path = {}
    for p in image_files:
        stem_to_path.setdefault(p.stem, p)
        stem_to_path.setdefault(normalize_mic_name(p.name), p)

    samples = []
    unmatched = []
    for mic_key, pts in points_by_mic.items():
        img_path = stem_to_path.get(mic_key)
        if img_path is None:
            unmatched.append(mic_key)
            continue
        samples.append((img_path, pts))

    if unmatched:
        print(f"WARNING: {len(unmatched)}/{len(points_by_mic)} micrograph(s) in {star_path} "
              f"have no matching file under {images_dir}, skipped. First few: {unmatched[:5]}",
              file=sys.stderr)
    if not samples:
        raise SystemExit(f"no micrograph under {images_dir} matched any entry in {star_path}")
    return samples


def split_train_val(samples, val_fraction, seed):
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if len(shuffled) > 1:
        n_val = max(1, min(n_val, len(shuffled) - 1))
    else:
        n_val = 0
    return shuffled[n_val:], shuffled[:n_val]


def subsample_train_mrcs(samples, num_mrcs, seed):
    """Randomly caps the training micrograph count; val_samples is untouched.

    Mirrors UPicker's --filter_num for its fine-tuning stage: UPicker's "Fine-tuning"
    section reports FT typically uses only 20-50 labeled images, far fewer than a full
    matched set -- --num_train_mrcs/--mrc_select_seed let a run reproduce that regime.
    """
    if num_mrcs is None or num_mrcs >= len(samples):
        return samples
    if num_mrcs <= 0:
        raise SystemExit(f'--num_train_mrcs must be positive, got {num_mrcs}')
    return random.Random(seed).sample(samples, num_mrcs)


def load_micrograph_rgb(img_path: Path):
    """Same grayscale->RGB conversion predict.py's infer() uses for mrc/jpg, so
    fine-tuning sees the identical preprocessing the model is later run against."""
    ext = img_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        orig_image = Image.open(img_path)
        w, h = orig_image.size
        gray_array = np.array(orig_image)
        rgb_array = np.repeat(gray_array[:, :, np.newaxis], 3, axis=2)
    elif ext == ".mrc":
        orig_image = denoise_micrographs.denoise(str(img_path))
        h, w = orig_image.shape
        rgb_array = np.repeat(orig_image[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError(f"unsupported micrograph extension: {img_path}")
    return Image.fromarray(rgb_array), w, h


def build_target(points, w, h, box_size, image_id):
    r = box_size / 2.0
    n = len(points)
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    for i, (x, y) in enumerate(points):
        boxes[i] = torch.tensor([x - r, y - r, x + r, y + r])
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]

    # label 0 = particle. There is no COCO category_id here to remap: this dataset
    # assigns the label directly, so the category_id=1 vs no-object=1 collision that
    # datasets/micrograph.py works around cannot occur through this path.
    labels = torch.zeros(boxes.shape[0], dtype=torch.int64)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)
    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([image_id]),
        "area": area,
        "iscrowd": iscrowd,
        "orig_size": torch.as_tensor([int(h), int(w)]),
        "size": torch.as_tensor([int(h), int(w)]),
    }


class StarMicrographDataset(torch.utils.data.Dataset):
    def __init__(self, samples, box_size, transforms):
        self.samples = samples
        self.box_size = box_size
        self._transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, points = self.samples[idx]
        rgb_image, w, h = load_micrograph_rgb(img_path)
        target = build_target(points, w, h, self.box_size, image_id=idx)
        img, target = self._transforms(rgb_image, target)
        return img, target


# ---------------------------------------------------------------------------
# finetune_mode -> trainable parameters
# ---------------------------------------------------------------------------

FINETUNE_MODES = (
    "head",
    "head_decoder",
    "head_decoder_encoder",
    "head_decoder_encoder_resnet",
)


def configure_finetune_mode(model, args):
    """Sets requires_grad per --finetune_mode.

    backbone.* is intentionally left untouched here: build_backbone() (models/backbone.py)
    already decides its trainability once, at build_model() time, from args.lr_backbone > 0
    (and always keeps resnet layer1 frozen regardless) -- main() sets args.lr_backbone to 0
    before build_model() for every mode except head_decoder_encoder_resnet.
    """
    # class_embed/bbox_embed (the prediction "head") are trainable in every mode: they
    # are what carries the from-scratch, collision-free label scheme, and are cheap
    # (class_embed is 256x2+2 params, bbox_embed a 3-layer MLP, ~132k).
    always_trainable = ("class_embed", "bbox_embed")
    extra_by_mode = {
        "head": (),
        "head_decoder": ("transformer.decoder", "query_embed"),
        "head_decoder_encoder": ("transformer.decoder", "query_embed",
                                  "transformer.encoder", "input_proj"),
        "head_decoder_encoder_resnet": ("transformer.decoder", "query_embed",
                                         "transformer.encoder", "input_proj"),
    }
    trainable_prefixes = always_trainable + extra_by_mode[args.finetune_mode]

    for name, p in model.named_parameters():
        if name.startswith("backbone"):
            continue
        p.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))


# ---------------------------------------------------------------------------
# validation (loss only -- see note in main() for why engine.evaluate isn't reused)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_losses(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    for samples, targets in metric_logger.log_every(data_loader, 10, 'Val:'):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
    metric_logger.synchronize_between_processes()
    print("Averaged val stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('CryoTransformer fine-tuning', add_help=False)

    # Experiment hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (UPicker fine-tuning stage default, '
                             'see module docstring; train.py uses 300 for full from-scratch training)')
    parser.add_argument('--backbone', default='resnet152', help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--device', default='cuda:0', help='CUDA device for GPU acceleration')
    parser.add_argument('--remarks', default='CryoTransformer_finetune', help='Additional remarks')
    parser.add_argument('--num_queries', type=int, default=600, help='Number of query slots')

    # Dataset: any GT-aligned STAR + a micrograph directory (see module docstring),
    # in place of train.py's COCO train_val_test_data.
    parser.add_argument('--images_dir', required=True, type=str,
                        help='Directory of .mrc/.jpg micrographs to fine-tune on')
    parser.add_argument('--star', required=True, type=str,
                        help='GT-aligned STAR (_rlnMicrographName/_rlnCoordinateX/_rlnCoordinateY) '
                             'covering the micrographs in --images_dir')
    parser.add_argument('--box_size', required=True, type=float,
                        help='Particle diameter in pixels, used to turn each STAR (x, y) center '
                             'into a square box for training (no diameter column in GT-aligned STAR)')
    parser.add_argument('--val_images_dir', default=None, type=str,
                        help='Optional separate validation micrograph dir; if omitted, val is a '
                             'micrograph-level split out of --images_dir/--star (see --val_fraction)')
    parser.add_argument('--val_star', default=None, type=str,
                        help='STAR paired with --val_images_dir; required if --val_images_dir is set')
    parser.add_argument('--val_fraction', default=0.1, type=float,
                        help='Held-out fraction of micrographs when --val_images_dir is not given')
    parser.add_argument('--num_train_mrcs', default=None, type=int,
                        help='Randomly cap how many training micrographs are used for fine-tuning '
                             '(the validation set is untouched); default uses every micrograph '
                             'matched from --images_dir/--star. Mirrors UPicker\'s fine-tuning '
                             'regime of 20-50 labeled images, see module docstring.')
    parser.add_argument('--mrc_select_seed', default=None, type=int,
                        help='Random seed for choosing which micrographs --num_train_mrcs keeps; '
                             'defaults to --seed when not given.')

    # Fine-tuning variant
    parser.add_argument('--finetune_mode', default='head', choices=FINETUNE_MODES,
                        help='Which parameters are trainable, see module docstring / '
                             'configure_finetune_mode()')

    # Output and resume paths
    parser.add_argument('--resume', default='pretrained_model/CryoTransformer_pretrained_model.pth',
                        help='Checkpoint to fine-tune from (the trained CryoTransformer, unlike '
                             "train.py's --resume default which bootstraps from generic DETR "
                             'weights before any micrograph-specific training has happened)')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Where to write checkpoints/logs; default is a timestamped dir '
                             'under output/finetuning/, mirroring train.py')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='Only used when --finetune_mode head_decoder_encoder_resnet; main() '
                             'forces this to 0 (frozen backbone) for every other mode')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=24, type=int,
                        help='StepLR decay epoch (UPicker fine-tuning stage default, see module '
                             'docstring; train.py uses 150 for full from-scratch training)')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset_file stays 'micrograph': models/detr.py's build() keys num_classes=1 off it
    parser.add_argument('--dataset_file', default='micrograph')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='run validation once and exit, no training')
    parser.add_argument('--eval_every', default=1, type=int, help='epochs between validation passes')
    parser.add_argument('--num_workers', default=2, type=int)

    # wandb: opt-in (train.py calls wandb.init() unconditionally; this script defaults it off
    # so unattended/offline fine-tuning runs -- e.g. smoke tests -- don't require a wandb login)
    parser.add_argument('--wandb', action='store_true', help='log to Weights & Biases')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.finetune_mode != 'head_decoder_encoder_resnet':
        # build_backbone() (models/backbone.py) decides backbone trainability once, at
        # construction time, from args.lr_backbone > 0 -- force it off here for every
        # mode that isn't the full-fine-tune one.
        args.lr_backbone = 0

    model, criterion, postprocessors = build_model(args)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # ###################################################################
        # #                                                                 #
        # #   --resume MUST BE A HEAD-REPAIRED CryoTransformer CHECKPOINT.   #
        # #                                                                 #
        # ###################################################################
        #
        # Every weight is loaded as-is here: class_embed and query_embed included,
        # nothing re-initialized. train.py drops those two and starts them fresh,
        # and so did this script until the repaired checkpoint existed.
        #
        # Do NOT point --resume at the RELEASED CryoTransformer weights. Their
        # class_embed is degenerate: training collapsed category_id=1 onto the
        # no-object index (the collision datasets/micrograph.py works around), so
        # the released head ranks queries no better than chance (AUC 0.531 on the
        # 22-ID held-out split, against a 0.601 positive rate). Fine-tuning on top
        # of that adapts a classifier that never learned to rank, and the fix is
        # not reachable from 40 micrographs: the repair needed 4,138.
        #
        # Use the head-repaired checkpoint of Sec. S2 (AUC 0.730 on the same split):
        #   $RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth
        # It is downloadable rather than reproducible from scratch -- see
        # src/rapick/picker/README.md for the Hugging Face path.
        #
        # Keeping its class_embed is only sound because the repaired head was
        # trained under the same 0 = particle scheme build_target() assigns, so
        # loaded head and fresh labels agree on which logit means "particle".
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        # A partial checkpoint is the failure this guards: it loads silently under
        # strict=False and leaves the head at its random init, which looks like a
        # fine-tuning run that simply did not converge.
        for name in ('class_embed.weight', 'class_embed.bias', 'query_embed.weight'):
            if name in missing:
                raise SystemExit(
                    f'{args.resume} has no {name}. Fine-tuning from it would train a '
                    f'randomly initialized head -- see the banner above and pass a '
                    f'head-repaired checkpoint.')
        if missing or unexpected:
            print(f'WARNING: --resume key mismatch: {len(missing)} missing, '
                  f'{len(unexpected)} unexpected. First few: {missing[:3]} / {unexpected[:3]}',
                  file=sys.stderr)

    # Must run before model.to(device)/optimizer construction, so param_dicts below
    # sees the final requires_grad flags.
    configure_finetune_mode(model, args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_total = sum(p.numel() for p in model_without_ddp.parameters())
    n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f'finetune_mode={args.finetune_mode}: {n_trainable:,} / {n_total:,} parameters '
          f'trainable ({100 * n_trainable / max(n_total, 1):.2f}%)')

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.val_images_dir:
        assert args.val_star, '--val_star is required together with --val_images_dir'
        train_samples = match_micrographs_to_star(Path(args.images_dir), Path(args.star))
        val_samples = match_micrographs_to_star(Path(args.val_images_dir), Path(args.val_star))
    else:
        all_samples = match_micrographs_to_star(Path(args.images_dir), Path(args.star))
        train_samples, val_samples = split_train_val(all_samples, args.val_fraction, args.seed)

    mrc_select_seed = args.mrc_select_seed if args.mrc_select_seed is not None else args.seed
    train_samples = subsample_train_mrcs(train_samples, args.num_train_mrcs, mrc_select_seed)
    print(f'{len(train_samples)} train / {len(val_samples)} val micrographs')

    dataset_train = StarMicrographDataset(train_samples, args.box_size, make_micrograph_transforms('train'))
    dataset_val = StarMicrographDataset(val_samples, args.box_size, make_micrograph_transforms('val'))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.eval:
        val_stats = evaluate_losses(model, criterion, data_loader_val, device)
        print(val_stats)
        return

    if args.wandb:
        import wandb
        wandb.init(name=args.remarks, config=vars(args))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Start fine-tuning")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device,
                                      epoch, args.clip_max_norm)
        lr_scheduler.step()

        checkpoint_paths = [output_path / 'checkpoint.pth']
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 20 == 0:
            checkpoint_paths.append(output_path / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_trainable}

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_stats = evaluate_losses(model, criterion, data_loader_val, device)
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

        if args.wandb:
            import wandb
            wandb.log(log_stats)

        if utils.is_main_process():
            with (output_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Fine-tuning time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CryoTransformer fine-tuning script', parents=[get_args_parser()])
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = "{}_mode_{}_timestamp_{}".format(args.remarks, args.finetune_mode, timestamp)
    output_dir = args.output_dir or "output/finetuning/{}".format(run_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
