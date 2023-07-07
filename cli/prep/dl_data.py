#!/usr/bin/env python3
# coding: utf-8
import os

import fire
import gdown

BASE_VQA2_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
BASE_COCO_URL = "http://images.cocodataset.org/zips/"


def download_vqa1_annotations(out_dir=None, download_train=True, download_val=True):
    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "Annotations_Train_mscoco.zip"
        train_url = BASE_VQA2_URL + train_file
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_val:
        val_file = "Annotations_Val_mscoco.zip"
        val_url = BASE_VQA2_URL + val_file
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)


def download_vqa1_questions(out_dir=None, download_train=True, download_val=True, download_test=True):

    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "Questions_Train_mscoco.zip"
        train_url = BASE_VQA2_URL + train_file
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_val:
        val_file = "Questions_Val_mscoco.zip"
        val_url = BASE_VQA2_URL + val_file
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)

    if download_test:
        test_file = "Questions_Test_mscoco.zip"
        test_url = BASE_VQA2_URL + test_file
        out_file = os.path.join(out_dir, test_file)
        gdown.cached_download(test_url, out_file, postprocess=gdown.extractall)


def download_vqa2_annotations(out_dir=None, download_train=True, download_val=True):

    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "v2_Annotations_Train_mscoco.zip"
        train_url = BASE_VQA2_URL + train_file
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_val:
        val_file = "v2_Annotations_Val_mscoco.zip"
        val_url = BASE_VQA2_URL + val_file
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)


def download_vqa2_questions(out_dir=None, download_train=True, download_val=True, download_test=True):

    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "v2_Questions_Train_mscoco.zip"
        train_url = BASE_VQA2_URL + train_file
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_val:
        val_file = "v2_Questions_Val_mscoco.zip"
        val_url = BASE_VQA2_URL + val_file
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)

    if download_test:
        test_file = "v2_Questions_Test_mscoco.zip"
        test_url = BASE_VQA2_URL + test_file
        out_file = os.path.join(out_dir, test_file)
        gdown.cached_download(test_url, out_file, postprocess=gdown.extractall)


def download_vqa2_images(out_dir=None, download_train=True, download_val=True, download_test=True):

    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "train2014.zip"
        train_url = BASE_COCO_URL + train_file
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_val:
        val_file = "val2014.zip"
        val_url = BASE_COCO_URL + val_file
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)

    if download_test:
        test_file = "test2015.zip"
        test_url = BASE_COCO_URL + test_file
        out_file = os.path.join(out_dir, test_file)
        gdown.cached_download(test_url, out_file, postprocess=gdown.extractall)


def download_nlvr2_images(base_url, out_dir=None, download_train=True, download_dev=True, download_test=True):
    if base_url is None:
        raise ValueError(
            "no base_url specified, please see https://github.com/lil-lab/nlvr/blob/master/nlvr2/README.md#direct-image-download to get base image url, it should be like https://xxx/NLVR2/"
        )

    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "train_img.zip"
        train_url = os.path.join(base_url, train_file)
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file, postprocess=gdown.extractall)

    if download_dev:
        val_file = "dev_img.zip"
        val_url = os.path.join(base_url, val_file)
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file, postprocess=gdown.extractall)

    if download_test:
        test_file = "test1_img.zip"
        test_url = os.path.join(base_url, test_file)
        out_file = os.path.join(out_dir, test_file)
        gdown.cached_download(test_url, out_file, postprocess=gdown.extractall)


def download_nlvr2_data(
    out_dir,
    download_train=True,
    download_dev=True,
    download_test=True,
    download_balance=False,
    download_unbalance=False,
):
    base_url = "https://github.com/lil-lab/nlvr/raw/master/nlvr2/data/"
    if out_dir is None:
        raise ValueError("no out_dir specified")

    if download_train:
        train_file = "train.json"
        train_url = os.path.join(base_url, train_file)
        out_file = os.path.join(out_dir, train_file)
        gdown.cached_download(train_url, out_file)

    if download_dev:
        val_file = "dev.json"
        val_url = os.path.join(base_url, val_file)
        out_file = os.path.join(out_dir, val_file)
        gdown.cached_download(val_url, out_file)

    if download_test:
        test_file = "test1.json"
        test_url = os.path.join(base_url, test_file)
        out_file = os.path.join(out_dir, test_file)
        gdown.cached_download(test_url, out_file)

    if download_balance:
        dev_file = "balanced/balanced_dev.json"
        val_url = os.path.join(base_url, dev_file)
        out_file = os.path.join(out_dir, "balanced_dev.json")
        gdown.cached_download(val_url, out_file)
        test_file = "balanced/balanced_test1.json"
        test_url = os.path.join(base_url, test_file)
        out_file = os.path.join(out_dir, "balanced_test1.json")
        gdown.cached_download(test_url, out_file)

    if download_unbalance:
        dev_file = "unbalanced/unbalanced_dev.json"
        val_url = os.path.join(base_url, dev_file)
        out_file = os.path.join(out_dir, "unbalanced_dev.json")
        gdown.cached_download(val_url, out_file)
        test_file = "unbalanced/unbalanced_test1.json"
        test_url = os.path.join(base_url, test_file)
        out_file = os.path.join(out_dir, "unbalanced_test1.json")
        gdown.cached_download(test_url, out_file)


if __name__ == "__main__":
    fire.Fire()
