from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()

import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
ASSETS_PATH = files("lib.assets")


@functools.lru_cache()
def _load_lines(path):
    """
    Load lines from a file with caching for performance.

    First tries to load from `path` directly. If that doesn't exist, searches the
    `lib.assets` directory for a file named `path`.

    Args:
        path: Path to the file (relative or absolute)

    Returns:
        List of stripped lines from the file
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or lib.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def from_file_eval(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    for prompt in prompts:
        yield prompt, {}


short_names = {
    "imagenet_all": "inall",
    "imagenet_animals": "inanm",
    "imagenet_dogs": "indog",
    "simple_animals": "simanm",
    "drawbench": "drawb",

    "hpd": "hpd",
    "hpd_photo": "hppho",
    "hpd_photo_painting": "hpphopa",
    "hpd_photo_anime": "hpphoan",
    "hpd_photo_concept": "hpphoct",

    "nouns_activities": "nounact",
    "counting": "count",
    "pap": "pap",
}

def pap():
    return from_file("pick_a_pic.txt")

def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def simple_animals_eval():
    return from_file_eval("simple_animals.txt")


import csv
import collections
@functools.lru_cache()
def read_csv(path):
    # reader = csv.DictReader(open(path))
    with open (path, 'r') as f:
        reader = csv.DictReader(f)
        reader = [row for row in reader]

    info = collections.defaultdict(list)
    for row in reader:
        info[row["Category"]].append(row["Prompts"])

    # Filter out 'Misspellings' and 'Rare Words' categories
    # Use only prompts [2:] from each category for training (first 2 reserved for eval)
    filtered_info = {}
    for k, v in info.items():
        if k in ["Misspellings", "Rare Words"]:
            continue
        filtered_info[k] = v[2:]
    drawbench_prompt_ls = sum(filtered_info.values(), [])
    return drawbench_prompt_ls

def drawbench():
    drawbench_prompt_ls = read_csv(ASSETS_PATH.joinpath("DrawBench Prompts.csv"))
    return random.choice(drawbench_prompt_ls), {}


import json
@functools.lru_cache()
def read_hpd(style=None):
    """
    Load HPDv2 prompts with caching.

    Args:
        style: Specific style to load ('anime', 'concept-art', 'paintings', 'photo')
               If None, loads all styles (~800 prompts each)

    Returns:
        List of prompt strings (first 10 from each style reserved for eval)
    """
    if style is None:
        styles = ["anime", "concept-art", "paintings", "photo"]
    else:
        styles = [style,]

    prompts_ls = []
    for style in styles:
        with open(ASSETS_PATH.joinpath(f"HPDv2/benchmark_{style}.json"), "r") as f:
            # Skip first 10 prompts (reserved for evaluation)
            prompts_ls.extend(json.load(f)[10:])

    return prompts_ls

def hpd():
    prompts_ls = read_hpd()
    return random.choice(prompts_ls), {}

def hpd_photo():
    prompts_ls = read_hpd("photo")
    return random.choice(prompts_ls), {}

def hpd_photo_painting():
    # Copy before extending, since the photo list is lru_cached.
    prompts_ls = read_hpd("photo") + read_hpd("paintings")
    return random.choice(prompts_ls), {}

def hpd_photo_anime():
    prompts_ls = read_hpd("photo") + read_hpd("anime")
    return random.choice(prompts_ls), {}

def hpd_photo_concept():
    prompts_ls = read_hpd("photo") + read_hpd("concept-art")
    return random.choice(prompts_ls), {}

def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata