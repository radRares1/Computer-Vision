import numpy as np
import cv2

if __name__ == '__main__':

    fn = 'flood.png'
    img = cv2.imread(fn, -1)

    h, w = img.shape[:2]
    kernel = np.zeros((h + 2, w + 2), np.uint8)
    seed_pt = None

    def update():
        if seed_pt is None:
            cv2.imshow('floodfill', img)
            return
        flooded = img.copy()
        kernel[:] = 0
        cv2.floodFill(flooded, kernel, seed_pt, (70, 202, 250), 0)
        cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
        cv2.imshow('floodfill', flooded)

    def onmouse(event, x, y, flags, param):
        global seed_pt
        if flags & cv2.EVENT_FLAG_LBUTTON:
            seed_pt = x, y
            update()

    update()
    cv2.setMouseCallback('floodfill', onmouse)

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        update()
    cv2.destroyAllWindows()