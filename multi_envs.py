import cv2
import numpy as np
from simulator import Simulator


if __name__ == '__main__':
    sim1 = Simulator('liberty.blend', points_density=10.0)
    sim2 = Simulator('liberty.blend', points_density=10.0)

    while True:

        key = cv2.waitKeyEx(7)

        if key == ord('q'):
            sim1.rotate_camera_yaw(-15)
            sim2.rotate_camera_yaw(15)
        elif key == ord('d'):
            sim1.rotate_camera_yaw(15)
            sim2.rotate_camera_yaw(-15)
        elif key == ord('z'):
            sim1.move_camera_forward(1)
            sim2.move_camera_forward(-1)
        elif key == ord('s'):
            sim1.move_camera_forward(-1)
            sim2.move_camera_forward(1)
        elif key == 27:  # escape key
            break

        rgb1, _ = sim1.render()
        rgb2, _ = sim2.render()

        cv2.imshow(f'rgb1', cv2.cvtColor(np.uint8(rgb1 * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'rgb2', cv2.cvtColor(np.uint8(rgb2 * 255), cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
