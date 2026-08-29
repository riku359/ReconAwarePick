# Modified copy of CryoTransformer's predict.py.
# Upstream: https://github.com/jianlin-cheng/CryoTransformer
# (MIT License, Copyright (c) 2023 Jianlin Cheng), pinned at commit
# a56f133f3f499562c32a6bc512eec5f115095b3b. The modification is in
# patches/predict.py.diff.
import denoise_micrographs
from glob import glob
import pandas as pd
import os
import csv
import cv2
import sys
import random
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.micrograph import make_micrograph_transforms

import matplotlib.pyplot as plt
import time


def nms(bounding_boxes, confidence_scores, nms_threshold, indices=None):
    # indices: optional array-like, same length as bounding_boxes, carrying an
    # external id (e.g. original query slot) for each box. When given, a third
    # return value maps each surviving box back to that id. Passing None keeps
    # the function's original 2-tuple return, so existing callers are unaffected.
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        if indices is not None:
            return [], [], []
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_indices = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_scores[index])
        if indices is not None:
            picked_indices.append(indices[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < nms_threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes).squeeze()
    picked_score = np.array(picked_score)

    if indices is not None:
        return picked_boxes, picked_score, np.array(picked_indices)
    return picked_boxes, picked_score

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                            img_w, img_h
                            ], dtype=torch.float32)
    return b

#changes by Ashwin
def get_images(in_path):  
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' or ext == '.mrc':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def resolve_test_data_root(cli_value):
    """Root holding <EMPIAR id>/images/, the directory this script reads micrographs from.

    Upstream hardcodes the relative path 'test_data/<id>/images', which on the authors'
    machines is a symlink into the data disk. The layout below the root is unchanged;
    only the root moved into the environment, so nothing points at one machine. Order:
    --data_root, then $RAPICK_TEST_DATA. There is no fallback -- a missing variable is
    an error naming it, never a default that happens to exist somewhere.
    """
    root = cli_value or os.environ.get('RAPICK_TEST_DATA')
    if not root:
        raise SystemExit(
            'RAPICK_TEST_DATA is not set; it must point at the root holding '
            '<EMPIAR id>/images/ (pass --data_root to override). '
            'See docs/CONFIGURATION.md.')
    return os.path.expanduser(root)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Test hyperparameters
    parser.add_argument('--quartile_threshold', type=float, default=0.25, help='Quartile threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.7, help='Non-maximum suppression threshold')
    parser.add_argument('--empiar', default='10081', help='EMPIAR ID for prediction')
    parser.add_argument('--data_root', default=None,
                        help='Root holding <EMPIAR id>/images/ with the micrographs to pick. '
                             'Defaults to $RAPICK_TEST_DATA; one of the two must be set.')
    parser.add_argument('--remarks', default='CryoTransformer_predictions', help='Additional remarks')
    parser.add_argument('--du_particles', default='N', choices=['Y', 'N'], help='DU Particles (Y or N)')
    parser.add_argument('--num_queries', type=int, default=600, help='Number of queries')
    parser.add_argument('--save_micrographs_with_encircled_proteins', default='Y', choices=['Y', 'N'], help='Plot predicted proteins on Micrographs (Y or N)')
    parser.add_argument('--gt-format', dest='gt_format', action='store_true',
                        help='Write the combined STAR in CryoPPP GT format: data_particles block, '
                             'columns _rlnMicrographName/_rlnCoordinateX/_rlnCoordinateY/'
                             '_rlnAutopickFigureOfMerit, integer top-left coords (Y un-flipped to '
                             'y=h-y_stored). Default is the native bottom-origin CryoSparc STAR.')
    parser.add_argument('--resume', default='pretrained_model/CryoTransformer_pretrained_model.pth', help='Resume path')
    parser.add_argument('--debug_dump', default=None,
                        help='[diagnosis-only, off by default] Directory to dump raw per-micrograph '
                             'inference tensors (pred_logits, pred_boxes, topk/keep, pre/post-NMS boxes '
                             'with original query indices) as .npz for offline hypothesis testing. Does '
                             'not alter picking/output behavior in any way.')
    parser.add_argument('--dump_hs', default=None,
                        help='[Stage-1 head repair, off by default] Directory to dump per-micrograph '
                             'decoder hidden states for offline class_embed retraining: main file '
                             '(<mic>.npz) has hs_last (600,256, fp16, class_embed input for the final '
                             'decoder layer), pred_logits (600,2, fp16, for the hs_last<->class_embed '
                             'sanity check), pred_boxes_px (600,4 cx,cy,w,h in ORIGINAL micrograph pixel '
                             'scale), query_idx (0..599) and mic_id; aux file (<mic>_hs_aux.npz) has '
                             'hs_layers_0_4 (5,600,256, fp16) for the intermediate decoder layers. '
                             'Independent of --debug_dump. Does not alter picking/output behavior.')
    parser.add_argument('--selection', default='legacy',
                        choices=['legacy', 'legacy_idxfix', 'softmax_topk', 'softmax_thresh'],
                        help="[diagnosis-only] Which query-selection route feeds NMS. 'legacy' (default) "
                             "is byte-identical to the original predict.py: exactly reproduces the "
                             "shipped, unmodified behavior. 'legacy_idxfix' keeps the same quantile-of-"
                             "sigmoid-topk mechanism and count, but fixes the index bug where `keep` "
                             "(a mask over score-sorted rank) is applied positionally to the raw "
                             "600-query array instead of being gathered via topk_indexes. 'softmax_topk' "
                             "selects the top --selection_topk_n queries by softmax particle-probability. "
                             "'softmax_thresh' selects queries with softmax particle-probability above "
                             "--selection_thresh.")
    parser.add_argument('--selection_topk_n', type=int, default=250,
                        help="query count for --selection softmax_topk")
    parser.add_argument('--selection_thresh', type=float, default=0.5,
                        help="probability threshold for --selection softmax_thresh")
  

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet152', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='micrograph')
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')


    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--thresh', default=0, type=float)   #edits by Ashwin, initially 0.5

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_dir):
    model.eval()
    duration = 0

    prefix_file_name = "EMPIAR_{}_remarks_{}".format(
    args.empiar, args.remarks
    )

    for img_sample in images_path:
        filename = os.path.basename(img_sample)[:-4] 
        print(len(filename))
        extension = img_sample[-3:]
        #loading image if input is in jpg format
        if extension == 'jpg':
            orig_image = Image.open(img_sample)
            img_size = orig_image.size
            # Broadcast the single intensity channel to R=G=B (grayscale -> RGB) via
            # numpy instead of a per-pixel Python loop over getdata()/putdata() -- same
            # ~38x speedup already applied to the mrc path below (see its comment); the
            # per-pixel loop here was previously untouched because prior work only ran
            # this codepath on small ad-hoc jpg batches, not a full 5172-image split.
            gray_array = np.array(orig_image)
            rgb_array = np.repeat(gray_array[:, :, np.newaxis], 3, axis=2)
            rgb_image = Image.fromarray(rgb_array)
            w, h = rgb_image.size

        if extension == 'mrc':
            orig_image = denoise_micrographs.denoise(img_sample)
            h, w = orig_image.shape
            # Broadcast the single intensity channel to R=G=B (grayscale -> RGB).
            # Verified pixel-identical to the original double for-loop across sample
            # micrographs before this replacement (analysis_diagnosis/, T5); ~38x faster
            # per micrograph (7.0-7.4s -> 0.18-0.20s on a 3838x3710 EMPIAR-10081 image).
            rgb_array = np.repeat(orig_image[:, :, np.newaxis], 3, axis=2)
            # Convert the NumPy array to a PIL Image
            rgb_image = Image.fromarray(rgb_array)

        transform = make_micrograph_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(rgb_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights, decoder_hidden = [], [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),
            # input[0] to class_embed is `hs`, shape (num_decoder_layers, B, Q, hidden_dim).
            # Keep the full stack (cheap: a few MB/image) so both the last-layer slice
            # (what pred_logits/pred_boxes are actually computed from, [-1]) and the
            # intermediate layers (--dump_hs's aux file) come from the same capture.
            model.class_embed.register_forward_hook(
                        lambda self, input, output: decoder_hidden.append(input[0].detach().cpu())

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        # print(outputs["pred_logits"])
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # print("=============probas softmax ===============================")
        # print(probas)

        probas2 = outputs['pred_logits'].sigmoid()
        topk_values, topk_indexes = torch.topk(probas2.view(outputs["pred_logits"].shape[0], -1), args.num_queries, dim=1)   #extreme important mention num queries

        if args.selection == 'legacy':
            # Unmodified upstream logic, kept byte-for-byte as the ground truth for
            # what predict.py has always shipped. Per diagnosis H1: `keep` is a mask
            # over score-sorted RANK (topk_values is sorted desc) but gets applied
            # positionally to the raw query array below — topk_indexes is computed
            # but never consulted, so this does not actually select the top-scoring
            # queries; it selects raw query slots 0..n_keep-1.
            scores = topk_values
            keep = scores[0] > np.quantile(scores, args.quartile_threshold)  #This is what prevents from predicting ice patches as particles
            scores = scores[0, keep]

            # keep = probas.max(-1).values > args.thresh  #this is original
            # print("==========" + img_sample + "====pred_logits after softmax===============================")
            # print(keep )

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], rgb_image.size)
            probas = probas[keep].cpu().data.numpy()
            selected_query_idx = np.nonzero(keep.cpu().numpy())[0]

        elif args.selection == 'legacy_idxfix':
            # Same quantile-of-sigmoid-topk mechanism and kept-count as legacy, but
            # actually gathers boxes/scores by the query id topk_indexes intended —
            # i.e. this is legacy with only the H1 index bug corrected, nothing else
            # changed (quartile_threshold, nms_threshold, num_queries all still apply).
            raw_rank_keep = topk_values[0] > np.quantile(topk_values, args.quartile_threshold)
            n_keep = int(raw_rank_keep.sum().item())
            kept_flat_idx = topk_indexes[0][:n_keep]           # score-sorted prefix, same n_keep as legacy
            kept_query_idx_sorted = (kept_flat_idx // 2).cpu().numpy()
            kept_scores_sorted = topk_values[0][:n_keep]
            # a query can occupy two flat slots (one per class logit, see H2); de-dup
            # keeping the higher-ranked (first) occurrence, preserving score-desc order
            _, first_pos = np.unique(kept_query_idx_sorted, return_index=True)
            order = np.sort(first_pos)
            selected_query_idx = kept_query_idx_sorted[order]
            scores = kept_scores_sorted[order]  # stays a torch tensor, like the legacy branch

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, selected_query_idx], rgb_image.size)
            probas = probas[selected_query_idx].cpu().data.numpy()

        elif args.selection == 'softmax_topk':
            # Select the top --selection_topk_n raw queries by softmax particle-
            # probability directly (the value H3 finds unused in legacy), bypassing
            # the sigmoid/topk/quantile machinery entirely.
            softmax_p0 = probas[:, 0]
            rank_order = torch.argsort(softmax_p0, descending=True)
            n_keep = min(args.selection_topk_n, softmax_p0.shape[0])
            selected_query_idx = rank_order[:n_keep].cpu().numpy()
            scores = softmax_p0[selected_query_idx]

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, selected_query_idx], rgb_image.size)
            probas = probas[selected_query_idx].cpu().data.numpy()

        elif args.selection == 'softmax_thresh':
            # Select raw queries whose softmax particle-probability exceeds an
            # absolute threshold, instead of a relative (quantile) cut.
            softmax_p0 = probas[:, 0]
            keep_mask = softmax_p0 > args.selection_thresh
            selected_query_idx = torch.nonzero(keep_mask, as_tuple=True)[0].cpu().numpy()
            scores = softmax_p0[keep_mask]

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep_mask], rgb_image.size)
            probas = probas[keep_mask].cpu().data.numpy()

        if args.debug_dump:
            # Boolean form over the raw 600-query array, for schema parity with the
            # legacy `keep` mask regardless of which route produced the selection.
            keep_np = np.zeros(outputs['pred_boxes'].shape[1], dtype=bool)
            keep_np[selected_query_idx] = True
            query_idx_in_original = selected_query_idx


        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        # h, w = conv_features['0'].tensors.shape[-2:]
        scores = scores.cpu().detach().numpy()
        if args.debug_dump:
            pre_nms_boxes_np = bboxes_scaled.cpu().numpy() if torch.is_tensor(bboxes_scaled) else np.array(bboxes_scaled)
            pre_nms_scores_np = scores.copy()
            boxes, scores, nms_survivor_query_idx = nms(
                bboxes_scaled, scores, nms_threshold=args.nms_threshold, indices=query_idx_in_original
            )
            os.makedirs(args.debug_dump, exist_ok=True)
            # nms()'s own `np.array(picked_boxes).squeeze()` yields an object array of
            # torch.Tensor when boxes is a list of 1-D tensors (numpy/torch interop quirk,
            # pre-existing in nms(), harmless for the .star path since that consumes boxes
            # element-by-element) — normalize to a plain float array here so the dump is
            # always loadable without allow_pickle.
            if len(boxes) == 0:
                nms_boxes_np = np.zeros((0, 4), dtype=np.float64)
            elif torch.is_tensor(boxes):
                nms_boxes_np = boxes.cpu().numpy()
            elif isinstance(boxes, np.ndarray) and boxes.dtype != object:
                nms_boxes_np = boxes
            else:
                nms_boxes_np = np.stack([
                    b.cpu().numpy() if torch.is_tensor(b) else np.asarray(b) for b in boxes
                ])
            np.savez(
                os.path.join(args.debug_dump, filename + '.npz'),
                pred_logits=outputs['pred_logits'][0].numpy(),          # (600, 2) raw logits
                pred_boxes=outputs['pred_boxes'][0].numpy(),            # (600, 4) normalized cxcywh
                topk_values=topk_values[0].cpu().numpy(),               # (600,) sigmoid top-k values, sorted desc
                topk_indexes=topk_indexes[0].cpu().numpy(),             # (600,) flat index into (query, class) pairs
                keep=keep_np,                                          # (600,) bool, positional mask (see H1)
                query_idx_in_original=query_idx_in_original,            # (n_kept,) raw query slot for each kept row
                pre_nms_boxes=pre_nms_boxes_np,                         # (n_kept, 4) xyxy, pixel scale
                pre_nms_scores=pre_nms_scores_np,                       # (n_kept,)
                nms_boxes=nms_boxes_np,                                 # (n_survive, 4) xyxy, pixel scale
                nms_scores=np.asarray(scores, dtype=np.float64),        # (n_survive,)
                nms_survivor_query_idx=nms_survivor_query_idx,           # (n_survive,) raw query slot, traced through NMS
                orig_w=w, orig_h=h,                                     # pre-resize micrograph size
                resized_hw=np.array(image.shape[-2:]),                   # post-transform (H, W) fed to the model
                hs=decoder_hidden[0][-1][0].numpy(),                    # (600, 256) class_embed input, last decoder layer
            )

        if args.dump_hs:
            os.makedirs(args.dump_hs, exist_ok=True)
            hs_all = decoder_hidden[0][:, 0]           # (6, 600, 256) fp32, all decoder layers, batch 0
            # normalized cxcywh -> pixel scale. Resize preserves aspect ratio, so a
            # fraction of the resized image is the same fraction of the original
            # micrograph.
            pred_boxes_px = outputs['pred_boxes'][0].numpy() * np.array([w, h, w, h], dtype=np.float32)
            np.savez(
                os.path.join(args.dump_hs, filename + '.npz'),
                hs_last=hs_all[-1].numpy().astype(np.float16),           # (600, 256)
                pred_logits=outputs['pred_logits'][0].numpy().astype(np.float16),  # (600, 2), for the hs<->class_embed sanity check
                pred_boxes_px=pred_boxes_px,                             # (600, 4) cx,cy,w,h, original micrograph pixel scale
                query_idx=np.arange(hs_all.shape[1], dtype=np.int16),    # (600,) raw query slot 0..599
                mic_id=filename,
                orig_w=w, orig_h=h,
            )
            np.savez(
                os.path.join(args.dump_hs, filename + '_hs_aux.npz'),
                hs_layers_0_4=hs_all[:-1].numpy().astype(np.float16),    # (5, 600, 256) intermediate decoder layers
            )

        if not args.debug_dump:
            boxes, scores = nms(bboxes_scaled, scores, nms_threshold=args.nms_threshold)
        print(f"----- generating star file for {filename}")
        # create directory for star files if not exist:
        box_file_path = output_dir + '/box_files/'
        predicted_particles_visualizations_path = output_dir + '/predicted_particles_visualizations/'
        if not os.path.exists(box_file_path):
            os.makedirs(box_file_path)
        if not os.path.exists(predicted_particles_visualizations_path):
            os.makedirs(predicted_particles_visualizations_path)
        save_individual_box_file(boxes, scores, img_sample, h, box_file_path, "_CryoTransformer")
        # print("=============boxes  ===============================")
        # print(boxes)
        # print("=============scores  ===============================")
        # print(scores)
        #edits by Ashwin
        if len(bboxes_scaled) == 0:
            print("there are no particle in image")
            continue

        if args.save_micrographs_with_encircled_proteins == 'Y':
            plot_predicted_boxes(rgb_image, boxes, filename, predicted_particles_visualizations_path, h)

        # print("=============== Predictions saved ===================")
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # infer_time = end_t - start_t
        # duration += infer_time
        # print("Processing END...{} ({:.3f}s)".format(filename, infer_time))

    # avg_duration = duration / len(images_path)

    # print("Avg. Time: {:.3f}s".format(avg_duration))

    #making header for combined star file:
    save_combined_star_file(box_file_path, prefix_file_name,
                            gt_format=args.gt_format, height=h)


def save_individual_box_file(boxes, scores, img_file, h, box_file_path, out_imgname):
    write_name = box_file_path + os.path.basename(img_file)[:-4] + out_imgname + '.box'
    with open(write_name, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow(["Micrograph_Name    X_Coordinate    Y_Coordinate    Class_Number    AnglePsi    Confidence_Score"])

        for i, box in enumerate(boxes):
            star_bbox = box.cpu().data.numpy()
            star_bbox = star_bbox.astype(np.int32)
            #h- is done to handle the cryoSparc micrograph reading orientation
            boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', (star_bbox[0] + star_bbox[2]) / 2, h-(star_bbox[1] + star_bbox[3]) / 2, -9999, -9999, scores[i]])
            if args.du_particles == 'Y':
                coordinate_shift_rand = random.choice(list(range(-20, -9)) + list(range(10, 21))) #shifting center to obtain better 2D averaging
                # coordinate_shift_rand = 10
                boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', ((star_bbox[0] + star_bbox[2]) / 2)+coordinate_shift_rand, (h-(star_bbox[1] + star_bbox[3]) / 2)+coordinate_shift_rand, -9999, -9999, scores[i]])

def plot_predicted_boxes(rgb_image, boxes, filename, predicted_particles_visualizations_path, h):
    img = np.array(rgb_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(boxes):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox_d = bbox.astype(np.int32)
        bbox_circle = bbox.astype(np.int32)


        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ])
        bbox = bbox.reshape((4, 2))
        # bbox_d = np.array([
        #     [bbox_d[0]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[3]+15],
        #     [bbox_d[0]+15, bbox_d[3]+15],
        #     ])
        # bbox_d = bbox_d.reshape((4, 2))


        bbox_circle_center = np.array([(bbox_circle[0] + bbox_circle[2]) / 2, (bbox_circle[1] + bbox_circle[3])/2]) #h- is ommitted here to handle the image plot
        bbox_circle_center = bbox_circle_center.reshape((1, 2))

        x_coordinate, y_coordinate = bbox_circle_center[0]
        center = (int(x_coordinate), int(y_coordinate))


        # cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
        # color=(0,255,0) #green
        color =(150, 255, 255) #purple
        radius=81
        thickness=10 # 7 earlier
        # cv2.polylines(img, [bbox_d], True, (0, 255, 0), 4)
        cv2.circle(img, center, radius, color, thickness)

    img_save_path = os.path.join(predicted_particles_visualizations_path, f"{filename}.jpg")

    cv2.imwrite(img_save_path, img)


def save_combined_star_file(box_file_path, prefix_file_name, gt_format=False, height=None):
    text_files = [file for file in os.listdir(box_file_path) if file.endswith('.box')]
    text_files.sort()
    output_file = output_dir + prefix_file_name + '_' + 'star_file.star'
    if gt_format:
        # The format the CryoPPP ground truth uses: data_particles, _rlnMicrographName,
        # integer top-left coordinates. The native Y is stored as h-Yc (bottom origin),
        # so Yc = height - Y_stored puts it back on a top-left origin.
        header = '''
data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnAutopickFigureOfMerit #4
'''
    else:
        header = '''
data_

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnClassNumber #4
_rlnAnglePsi #5
_rlnAutopickFigureOfMerit #6
'''

    with open(output_file, 'w') as outfile:
        # Write the header content to the new file
        outfile.write(header)

        # Iterate over each text file
        for file in text_files:
            # Open the current file in read mode
            with open(os.path.join(box_file_path, file), 'r') as infile:
                # Skip the first line
                next(infile)
                if not gt_format:
                    # Read the remaining content of the file (native format, unchanged)
                    outfile.write(infile.read())
                    continue
                # GT format: reformat each row (drop Class/Psi, flip Y, integer coords)
                for line in infile:
                    t = line.rstrip('\n').split('\t')
                    if len(t) < 6:
                        continue
                    mic, x, y_stored, _cls, _psi, fom = t[0], t[1], t[2], t[3], t[4], t[5]
                    xi = int(round(float(x)))
                    yi = int(round(height - float(y_stored)))
                    outfile.write(f"{mic} {xi} {yi} {fom}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CryoTransformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    from datetime import datetime
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    data_path = os.path.join(resolve_test_data_root(args.data_root),
                             str(args.empiar), "images")  #cryoPPP ~300 micrographs
    output_dir = "output/predictions/predictions_EMPIAR_{}_remarks_{}_timestamp_{}/".format(
    args.empiar, args.remarks, timestamp)


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(data_path)
    print(image_paths)

    infer(image_paths, model, postprocessors, device, output_dir)