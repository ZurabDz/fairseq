#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random

# Adding support for mp3 files
import torchaudio
# Adding paralellism and progress bar
import mpire 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing mp3 files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser

def get_audio_frames(fname: str) -> int:
   sig, sr = torchaudio.load(fname)

   # FIXME: This might not be an int and dont wanna // right now
   return {'fname': fname, 'n_frames': sig.shape[-1]}  

def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    fnames = list(glob.iglob(search_path, recursive=True))

    with mpire.WorkerPool(n_jobs=6) as pool:
        results = pool.map(get_audio_frames, fnames[:80_000], progress_bar=True)

    random.shuffle(results)

    valid_entries = results[:int(len(results) * args.valid_percent)] 
    train_entries = results[int(len(results) * args.valid_percent):]
    
    valid_f = (
        open(os.path.join(args.dest, "valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )

    train_f = open(os.path.join(args.dest, "train.tsv"), "w") 

    for processed_entry in valid_entries:
        print(f"{processed_entry['fname']}\t{processed_entry['n_frames']}", file=valid_f)

    for processed_entry in train_entries:
        print(f"{processed_entry['fname']}\t{processed_entry['n_frames']}", file=train_f)

    if valid_f is not None:
        valid_f.close()
    train_f.close()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
