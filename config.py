from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Config:
    # Experiment name. All data artifacts go under cache/{NAME}/ and all
    # training checkpoints go under runs/{NAME}/, so one knob isolates a
    # whole run from any other.
    NAME: str = "default"

    # data
    N: int = 1000
    K: int = 100
    SEQ_LEN: int = 512
    SEED : int = 0
    SHUFFLE_SEED : int = 1

    # Which attributes appear in each bio. Subset of data.bio_text.FIELD_SPECS
    # keys ("birthday", "birthcity", "university", "field", "company_city",
    # "company_name"). Default is birthday-only; flip to all six to reproduce
    # the legacy Capo bioS layout.
    FIELDS: tuple = ("birthday",)

    PRE_REDUCE_PATH: str = "cache/default/bios_prereduce.bin"
    POST_REDUCE_PATH: str = "cache/default/bios_postreduce.bin"

    # model
    MODEL_TYPE: str = "llama"
    numLayers: int = 8
    dmodel: int = 512
    numHeads: int = 8

    # vocab
    vocab_size : int= 512
    eosToken : int = 50257 #updated after remapping

    reducedVocabSize : int = 100 #dummy values
    reducedEOSToken : int = 100 #dummy values
    #
    BATCH_SIZE : int = 24
    LR : float = 5e-4
    WEIGHT_DECAY : float = 0.01
    WARMUP_STEPS : int= 200
    MAX_STEPS : int= 5000
    GRAD_CLIP : float = 1.0
    # -------------------
    # save
    # -------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    # -------------------
    # load
    # -------------------
    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)