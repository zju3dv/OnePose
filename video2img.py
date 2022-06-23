import os.path as osp
from argparse import ArgumentParser
from pathlib import Path
from src.utils.data_utils import video2img


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="The video file or directory to be parsed")
    parser.add_argument("--downsample", default=1, type=int)
    args = parser.parse_args()

    input = args.input
    
    if osp.isdir(input): # in case of directory which contains video file
        video_file = osp.join(input, 'Frames.m4v')
    else: # in case of video file
        video_file = input
    assert osp.exists(video_file), "Please input an valid video file!"

    data_dir = osp.dirname(video_file)
    out_dir = osp.join(data_dir, 'color_full')
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    video2img(video_file, out_dir, args.downsample)


if __name__ == "__main__":
    main()