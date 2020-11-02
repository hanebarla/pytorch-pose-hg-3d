import cv2
import os


def capture_img():
    camera = cv2.VideoCapture(0)
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


if __name__ == "__main__":
    capture_img()
