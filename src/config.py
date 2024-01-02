from dataclasses import dataclass
from simple_parsing import ArgumentParser
from pathlib import Path


__repo_path__ = Path(__file__).parent.parent
__data_path__ = __repo_path__ / 'data'


@dataclass
class OptionsCfg:
    """ Help string for basic options """
    example: str = '02_160.mov'  # source video path


# -------- Config -------- #


@dataclass
class Config:
    # basic options
    opts: OptionsCfg = OptionsCfg()


def get_cfg() -> Config:
    # parse config options
    parser = ArgumentParser()
    parser.add_arguments(OptionsCfg, dest="opts")
    args = parser.parse_args()

    # return config object
    return Config(
        opts=args.opts,
    )
