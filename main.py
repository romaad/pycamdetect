import argparse
import threading
from configparser import ConfigParser
from dataclasses import dataclass
import random
import sys
from typing import Any, List
import cv2
from datetime import datetime
import signal
import csv
from pathlib import Path
from os.path import exists


is_loop = True
CONF_PATH = "/etc/.pycam.conf"


@dataclass
class CamConfig:
    gui: bool = False
    path: str = ""
    name: str = ""
    refresh: int = 0


def start_loop(csv_writer, conf: CamConfig):
    global is_loop
    initial_state = None
    video = cv2.VideoCapture(get_cam_source(conf.path))

    # starting the webCam to capture the video using cv2 module
    start_time = datetime.now()
    is_moving = False
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    video_writer: Any = None
    motion_start_time = datetime.now()

    if not video.isOpened():
        print(
            f"cam: {conf.name} in location: {conf.path} isn't available",
            file=sys.stderr,
        )
        return

    # using infinite loop to capture the frames from the video
    while is_loop:
        _, cur_frame = video.read()
        var_motion = 0

        # From colour images creating a gray frame
        gray_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        # To find the changes creating a GaussianBlur from the gray scale image
        gray_frame = cv2.GaussianBlur(gray_image, (21, 21), 0)

        cur_time = datetime.now()

        if initial_state is None or (cur_time - start_time).seconds > conf.refresh:
            initial_state = gray_frame
            start_time = cur_time
            continue

        # Calculation of difference between static or initial and gray frame we created
        differ_frame = cv2.absdiff(initial_state, gray_frame)

        # the change between static or initial background and current gray frame are highlighted
        thresh_frame = cv2.threshold(
            differ_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        # For the moving object in the frame finding the coutours
        cont, _ = cv2.findContours(
            thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cur in cont:
            if cv2.contourArea(cur) < 70:
                continue
            var_motion = True
            (cur_x, cur_y, cur_w, cur_h) = cv2.boundingRect(cur)
            # To create a rectangle of green color around the moving object
            cv2.rectangle(cur_frame, (cur_x, cur_y),
                          (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 1)

        # Adding the Start time of the motion
        if not is_moving and var_motion:
            motion_start_time = datetime.now()
            # store a still for the first frame of movement
            write_cap(conf.name, cur_frame)
            # start storing video for movement
            video_writer = get_video_handler(
                conf.name, frame_width, frame_height)

        # Adding the End time of the motion
        elif is_moving and not var_motion:
            write_to_csv(csv_writer, motion_start_time, datetime.now())
            video_writer.release()

        if var_motion:
            video_writer.write(cur_frame)

        is_moving = var_motion

        if conf.gui:
            cv2.imshow(
                f"{conf.name}: stream", cur_frame)
            cv2.waitKey(1)
    if video:
        video.release()
    if video_writer:
        video_writer.release()


def write_cap(name: str, frame):
    now = datetime.now()
    img_dir = f"caps/{name}/{now.year}/{now.month}/{now.day}/"
    path = Path(img_dir)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        f"{img_dir}/{int(now.timestamp())}.jpg",
        frame,
    )


def get_video_handler(name: str, w: int, h: int):
    now = datetime.now()
    img_dir = f"videos/{name}/{now.year}/{now.month}/{now.day}/"
    path = Path(img_dir)
    path.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(f"{img_dir}/v{int(now.timestamp())}.mp4",
                           cv2.VideoWriter_fourcc(*"MP4V"), 20.0, (w, h))


def SignalHandler_SIGINT(sig_num, data):
    global is_loop
    print(f'SignalHandler of signal.SIGINT {sig_num}, {data}')
    is_loop = False


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def init_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--refresh", help="refresh period in seconds", type=int)
    parser.add_argument("--name", help="camera name for storage", type=int)
    parser.add_argument("--gui", help="os supports GUI", action="store_true")
    parser.add_argument("--path", default="0",
                        help="camera dev id or stream url", type=str)
    return parser.parse_args()


def write_to_csv(fcsv_writer: csv.DictWriter, start: datetime, end: datetime):
    fcsv_writer.writerow(
        {"Initial": start, "Final": end})


def get_cam_source(inp: str):
    try:
        return int(inp)
    except ValueError:
        return inp


def read_conf() -> List[CamConfig] | None:
    if not exists(CONF_PATH):
        print("can't find config file", file=sys.stderr)
        return None
    conf = ConfigParser()
    conf.read(CONF_PATH)
    gui = str2bool(conf["system"]["gui"])
    refresh = int(conf["system"]["refresh"])
    ret: List[CamConfig] = []
    for key in conf.sections():
        if (key != "system"):
            ret.append(CamConfig(
                gui=gui, refresh=refresh,
                name=conf[key]["name"],
                path=conf[key]["path"],
            ))

    print("found config")
    print(ret)

    return ret


parser = argparse.ArgumentParser(
    prog='cam_motion_capture',
    description='This program captures the video stream and detects movement and records it locally',
    epilog='By Mohamed Said, GPL',
)


def main():
    args = init_parser(parser)

    # register ctrl+c sigint handler
    signal.signal(signal.SIGINT, SignalHandler_SIGINT)
    signal.signal(signal.SIGTERM, SignalHandler_SIGINT)
    # List of all the tracks when there is any detected of motion in the frames
    fcsv = open("motion.csv", "a")
    csv_writer = csv.DictWriter(fcsv, ["Initial", "Final"])

    random.seed()
    conf = read_conf()
    if not conf:
        conf = [CamConfig(args.gui, args.path, args.name, args.refresh)]
    # main loop
    threads = [threading.Thread(target=start_loop, args=(
        csv_writer, item)) for item in conf]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    fcsv.close()
    # Now, Closing or destroying all the open windows with the help of openCV
    cv2.destroyAllWindows()
    exit(0)


main()
