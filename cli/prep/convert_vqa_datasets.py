#!/usr/bin/env python3
# coding: utf-8
import json
import os
import re
from pathlib import Path

import fire

"""
modified from https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/392520640e3d9aed0009ddfe207901757b10b9a6/dataset.py
"""

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


ans_field = {
    "vqa1": "multiple_choice_answer",
    "vqa2": "multiple_choice_answer",
    "vg": "answer",
    "vizwiz": "answers",
}
q_id_field = {
    "vqa1": "question_id",
    "vqa2": "question_id",
    "vg": "qa_id",
    "vizwiz": "image",
}
img_id_field = {
    "vqa1": "image_id",
    "vqa2": "image_id",
    "vg": "image_id",
    "vizwiz": "image",
}


def filter_vw_answers(qa_data, min_occurence, dataset="vw"):
    """This will change the answer to preprocessed version"""
    occurence = {}

    for ans_entry in qa_data:
        answers = ans_entry["answers"]
        for ans in answers:
            gtruth = ans["answer"]
            gtruth = preprocess_answer(gtruth)
            if gtruth not in occurence:
                occurence[gtruth] = set()
            occurence[gtruth].add(ans_entry["image"])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            # print('popping', answer)
            occurence.pop(answer)

    print("Num of answers that appear >= %d times: %d" % (min_occurence, len(occurence)))
    return occurence


def filter_answers(qa_data, min_occurence, dataset="vqa2"):
    """This will change the answer to preprocessed version"""
    occurence = {}

    for ans_entry in qa_data:
        # answers = ans_entry['answers']
        gtruth = ans_entry[ans_field[dataset]]
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry[q_id_field[dataset]])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            # print('popping', answer)
            occurence.pop(answer)

    print("Num of answers that appear >= %d times: %d" % (min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurrence, name, out_dir):
    ans2label = {}
    label2ans = []
    label = 0
    for answer in sorted(occurrence):
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    os.makedirs(out_dir, exist_ok=True)

    cache_file = os.path.join(out_dir, name + "_ans2label.json")
    json.dump(ans2label, open(cache_file, "w"), indent=2)
    cache_file = os.path.join(out_dir, name + "_label2ans.json")
    json.dump(label2ans, open(cache_file, "w"), indent=2)
    return ans2label


def compute_target(qa_data, id_questions, ans2label, out_dir, dataset="vqa2", split="val2014"):
    with open(f"{out_dir}/{dataset}-{split}.jsonl", "w") as f:
        for entry in qa_data:
            labels = None

            if dataset == "vg":
                ans = entry.get("answer", None)
                if ans:
                    ans_ = preprocess_answer(ans)
                    # single answer
                    labels = {ans_: 1 if ans_ in ans2label else 0}
            elif dataset in ["vqa1", "vqa2", "vizwiz"]:
                answers = entry.get("answers", None)
                if answers:
                    answer_count = {}
                    for answer in answers:
                        answer_ = answer["answer"]
                        answer_count[answer_] = answer_count.get(answer_, 0) + 1

                    labels = {}
                    for answer in answer_count:
                        if answer not in ans2label:
                            continue
                        score = get_score(answer_count[answer])
                        labels[answer] = score
            else:
                raise ValueError("not supported dataset type: {}".format(dataset))

            qid = entry[q_id_field[dataset]]
            img_id = entry[img_id_field[dataset]]
            if dataset in ["vqa1", "vqa2"]:
                q_sent = id_questions[qid]
                if split.startswith("test"):
                    split = "test2015"
                image_id = "COCO_{}_{:012d}".format(split, img_id)
            else:  # vg, vizwiz or no id
                q_sent = entry["question"]
                image_id = img_id

            target = {
                "qid": qid,
                "question": q_sent,
                "img_id": image_id,
            }
            if labels is not None:  # caveat! label can be empty
                target["label"] = labels
            ans_type = entry.get("answer_type", "")
            if ans_type:
                target["answer_type"] = ans_type
            f.write(json.dumps(target, ensure_ascii=False))
            f.write("\n")


def process_vqa(dataset_dir, out_dir=None, version=1):
    if out_dir is None:
        out_dir = dataset_dir
    if version == 1:
        prefix = ""
        name = "vqa1"
    else:
        prefix = "v2_"
        name = "vqa2"
    train_answer_file = f"{dataset_dir}/{prefix}mscoco_train2014_annotations.json"
    train_answers = json.load(open(train_answer_file))["annotations"]
    train_question_file = f"{dataset_dir}/{prefix}OpenEnded_mscoco_train2014_questions.json"
    train_questions = json.load(open(train_question_file))["questions"]
    train_id_questions = {i["question_id"]: i["question"] for i in train_questions}
    print("loaded coco train2014")
    val_answer_file = f"{dataset_dir}/{prefix}mscoco_val2014_annotations.json"
    val_answers = json.load(open(val_answer_file))["annotations"]
    val_question_file = f"{dataset_dir}/{prefix}OpenEnded_mscoco_val2014_questions.json"
    val_questions = json.load(open(val_question_file))["questions"]
    val_id_questions = {i["question_id"]: i["question"] for i in val_questions}
    print("loaded coco val2014")

    train_val_answers = train_answers + val_answers
    train_val_occurrence = filter_answers(train_val_answers, 9, name)
    trainval_ans2label = create_ans2label(train_val_occurrence, name, out_dir=out_dir)

    test_question_file = f"{dataset_dir}/{prefix}OpenEnded_mscoco_test2015_questions.json"
    test_questions = json.load(open(test_question_file))["questions"]
    print("loaded coco test2015")

    test_dev2015_question_file = f"{dataset_dir}/{prefix}OpenEnded_mscoco_test-dev2015_questions.json"
    test_dev2015_questions = json.load(open(test_dev2015_question_file))["questions"]
    print("loaded coco test-dev2015")

    cache_root = "data/cache"
    os.makedirs(cache_root, exist_ok=True)
    compute_target(
        train_answers,
        train_id_questions,
        trainval_ans2label,
        dataset=name,
        split="train2014",
        out_dir=out_dir,
    )
    print("saved coco train2014")
    compute_target(
        val_answers,
        val_id_questions,
        trainval_ans2label,
        dataset=name,
        split="val2014",
        out_dir=out_dir,
    )
    print("saved coco val2014")
    test_id_questions = {i["question_id"]: i["question"] for i in test_questions}
    compute_target(test_questions, test_id_questions, {}, dataset=name, split="test2015", out_dir=out_dir)
    print("saved coco test2015")
    test_dev2015_id_questions = {i["question_id"]: i["question"] for i in test_dev2015_questions}
    compute_target(
        test_dev2015_questions, test_dev2015_id_questions, {}, dataset=name, split="test-dev2015", out_dir=out_dir
    )
    print("saved coco test-dev2015")


def process_vg(dataset_dir, out_dir=None):
    if out_dir is None:
        out_dir = dataset_dir
    # process vg qa data
    vg_qa_file = f"{dataset_dir}/question_answers.json"
    vg_qa = json.load(open(vg_qa_file))
    print("loaded vg")
    vg_answers = [qai for qas in vg_qa for qai in qas["qas"]]
    vg_occurrence = filter_answers(vg_answers, 9, dataset="vg")
    # pickle.dump(vg_occr, open('data/cache/vg_occur.pk', 'wb'))
    vg_ans2label = create_ans2label(vg_occurrence, "vg", out_dir=out_dir)
    compute_target(vg_answers, {}, vg_ans2label, dataset="vg", split="all", out_dir=out_dir)
    print("saved vg, all done")


def process_vlue_vqa(dataset_dir):
    dataset_dir = Path(dataset_dir)
    vlue_qa_file = dataset_dir / "vqa_vlue_test.json"
    vlue_qa = json.load(open(vlue_qa_file))
    print("loaded vlue vqa")
    with open(vlue_qa_file.with_suffix(".jsonl"), "w") as f:
        for entry in vlue_qa:
            target = {
                "qid": entry["question_id"],
                "question": entry["question"],
                "img_id": entry["image"],
                "label": {entry["answer"]: 1},
            }
            f.write(json.dumps(target, ensure_ascii=False))
            f.write("\n")
    print("all vlue vqa converted")


def process_vizwiz(dataset_dir, out_dir=None):
    if out_dir is None:
        out_dir = dataset_dir
    # process vizwiz qa data
    vw_file = f"{dataset_dir}/Annotations/train.json"
    vw_train_qa = json.load(open(vw_file))
    print("loaded vizwiz train")

    vw_val_file = f"{dataset_dir}/Annotations/val.json"
    vw_val_qa = json.load(open(vw_val_file))
    print("loaded vizwiz val")

    vw_qa = vw_train_qa + vw_val_qa
    vw_occurrence = filter_vw_answers(vw_qa, 5, dataset="vizwiz")
    vw_ans2label = create_ans2label(vw_occurrence, "vizwiz", out_dir=out_dir)
    compute_target(vw_train_qa, {}, vw_ans2label, dataset="vizwiz", split="train", out_dir=out_dir)
    print("saved vizwiz train")

    compute_target(vw_val_qa, {}, vw_ans2label, dataset="vizwiz", split="val", out_dir=out_dir)
    print("saved vizwiz val")

    vw_test_file = f"{dataset_dir}/Annotations/test.json"
    vw_test_qa = json.load(open(vw_test_file))
    print("loaded vizwiz test")
    compute_target(vw_test_qa, {}, vw_ans2label, dataset="vizwiz", split="test", out_dir=out_dir)
    print("saved vizwiz test, all done")


if __name__ == "__main__":
    fire.Fire()
