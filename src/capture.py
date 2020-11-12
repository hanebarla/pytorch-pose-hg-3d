import cv2
import os
import argparse
import time


FPS = 60


def Img(camera):
    loop_on = 1
    count = 0

    while loop_on:
        ret, frame = camera.read()
        cv2.imshow('img', frame)

        if count > 70 and ((count % 10) == 0):
            img_path = os.path.join("images",
                                    "calb_img",
                                    "img_for_calib_{}.png".format(count))
            # print(img_path)
            cv2.imwrite(img_path, frame)

        count += 1
        k = cv2.waitKey(100)
        if k == 27:
            loop_on = 0

    cv2.destroyAllWindows()
    camera.release()


def Video(camera):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FPS, FPS)
    fps = camera.get(cv2.CAP_PROP_FPS)
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps)

    Video_path = os.path.join("images", "calb_img", "video_for_calib.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(Video_path, fourcc, fps, (w, h))
    loop_on = 1
    count = 0

    starttime = time.time()
    while loop_on:
        ret, frame = camera.read()

        if ret:
            cv2.imshow('frame', frame)
            out.write(frame)
        count += 1

        k = cv2.waitKey(1)
        if k == 27:
            loop_on = 0

    totaltime = time.time() - starttime
    print(totaltime)
    camera.release()
    cv2.destroyAllWindows()


def capture_img():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["image", "video"], default="image")
    args = parser.parse_args()
    CapMode = args.mode
    camera = cv2.VideoCapture(0)

    if CapMode == "image":
        Img(camera)
    if CapMode == "video":
        Video(camera)


if __name__ == "__main__":
    capture_img()
