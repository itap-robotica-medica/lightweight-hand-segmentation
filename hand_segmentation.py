import mediapipe as mp
import cv2
import time
import os
import tqdm
import numpy as np
import hands_connections
from sklearn.metrics import jaccard_score


def extract_hands(image):
    """
    :return: Tuple of:
    Black image with the skeleton drawn,
    Array with x, y position of all landmarks,
    Message with 21 * 4 + 2 elements where the two first elements can be "L", "R" or -100.
    These two first elements signal the order of the handedness in the following elements. The next
    84 elements are integers signaling the x,y coordinates in screen space (pixels) of each one
    of the 21 landmark points of each hand. For example, if the message is R, L, x1, y1, ...
    it means that the first 42 integers will correspond to the right hand while the last 42 will be
    the positions of the left hand landmarks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    black_canvas = np.zeros_like(image, dtype=np.uint8)
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # https://google.github.io/mediapipe/solutions/hands.html#output
    if results.multi_hand_landmarks:
        # Se almacena uno por cada mano: len(results.multi_hand_landmarks) = 1 o 2
        for hi, (handedness, hand) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
            mp_drawing.draw_landmarks(
                black_canvas,
                hand,
                hands_connections.HAND_CONNECTIONS,
                mp_drawing_styles.get_custom_hand_connections_style())
            mp_drawing.draw_landmarks(out, hand, hands_connections.HAND_CONNECTIONS, mp_drawing_styles.get_custom_hand_connections_style())
    return black_canvas, out


def binaryMaskIOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


def list_RGB2BGR(a):
    a[0], a[2] = a[2], a[0]
    return a


def draw_mask(image, mask, draw_zero=False, alpha=1.0, colors_dict=None):
    """
    colors_dict overwrites alpha if colors have 4 values and draw zero is a key "0" is in the dict.
    alpha can be overwritten by colors with 4 values.
    Add annotations
    """
    img = image.copy()
    # Case 1: Array of masks

    # Case 2: Image with a different pixel value for each mask
    assert image.shape
    unique_values = np.unique(mask)
    if True:
        # Convert colors to BGR and convert them to np.array
        colors_dict = {k: np.array(list_RGB2BGR(v)) for k, v in colors_dict.items()}
        if "0" in colors_dict:
            draw_zero = True
    for i, color in colors_dict.items():
        if draw_zero or i != 0:
            img_mask = mask[:, :, 0] == int(i)
            i_alpha = alpha
            if len(color) == 4:
                i_alpha = color[-1] / 255
            color = img[img_mask] * (1 - i_alpha) + color[:3] * i_alpha
            img[img_mask] = color
    return img


if __name__ == "__main__":
    # Mediapipe settings
    mp_hands = mp.solutions.hands
    import custom_drawing
    mp_drawing = custom_drawing
    mp_drawing_styles = custom_drawing


    # Open mediapipe process
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2, static_image_mode=False) as hands:

        ious = []
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        kernel_dilation_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))  # np.ones((9, 9), np.uint8)
        kernel_dilation2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # np.ones((3, 3), np.uint8)
      
        for dir in [f"Ego2Hands_eval/eval_seq{i}_imgs/" for i in range(8, 9)]:

            dir_ious = []
            image_ious = []

            print(dir)

            for image_path in tqdm.tqdm([f for f in os.listdir(dir) if len(f) == 9]):  # ["hand_mini.jpg", "00000.png", "00002.png", "00028.png", "00030.png", "00040.png", "00044.png", "00064.png", "00112.png", "00128.png"]:

                image_path = dir + image_path
                mask_path = image_path[:-4] + "_seg.png"

                image = cv2.imread(image_path)
                # image = cv2.resize(image, (400, 224))
                ground_truth = cv2.imread(mask_path)
                # ground_truth = cv2.resize(ground_truth, (400, 224))
                _, ground_truth = cv2.threshold(ground_truth, 1, 255, cv2.THRESH_BINARY)

                acc_mp = 0
                acc_morp = 0
                acc_ranges = 0
                acc_cvt = 0
                t0 = time.time()

                iters = 100

                for i in range(iters):

                    # Stage 1: Get landmark mask from Mediapipe
                    skeleton, out = extract_hands(image)
                    mp_0 = time.time()
                    # skeleton = extract_hands(image)
                    acc_mp += (time.time() - mp_0)

                    cvt_0 = time.time()
                    skeleton_gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
                    _, skeleton_binary = cv2.threshold(skeleton_gray, 1, 255, cv2.THRESH_BINARY)
                    acc_cvt += (time.time() - cvt_0)

                    # Stage 2: Get hand zone
                    morp_0 = time.time()
                    skeleton_binary = cv2.dilate(skeleton_binary, kernel_dilation_mask, iterations=1)
                    mask_s2 = cv2.dilate(skeleton_binary, kernel_dilation, iterations=2)
                    acc_morp += (time.time() - morp_0)

                    error = False

                    try:
                        # Convert image to hsv
                        cvt_0 = time.time()
                        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                        acc_cvt += (time.time() - cvt_0)

                        ranges_0 = time.time()
                        # Get boolean mask of interest pixels
                        mask_landmarks = skeleton_binary == 255
                        # Get hue, saturation and value values in previously selected pixels
                        masked_hsv = np.where(mask_landmarks != 0)
                        masked_hue = hsv_image[:, :, 0][masked_hsv]
                        masked_sat = hsv_image[:, :, 1][masked_hsv]
                        masked_val = hsv_image[:, :, 2][masked_hsv]
                        # Get percentiles of each channel values
                        # This could be further optimized if needed getting the random choice out of only one array
                        n_samples = 150
                        # q1h, q3h = np.percentile(np.random.choice(masked_hue, n_samples), [1, 99])      # 1, 99 (CIE)   40, 60 (hsv)
                        q1s, q3s = np.percentile(np.random.choice(masked_sat, n_samples), [25, 75])      # 25, 75        2, 98
                        q1v, q3v = np.percentile(np.random.choice(masked_val, n_samples), [25, 75])      # 25, 75        2, 98

                        # Filter image based on calculated percentiles
                        factorh = 0.0
                        factors = 0.025
                        factorv = 0.025
                        # mask_hue = cv2.inRange(hsv_image[:, :, 0], int(q1h * (1 - factorh)), int(q3h * (1 + factorh)))
                        mask_sat = cv2.inRange(hsv_image[:, :, 1], int(q1s * (1 - factors)), int(q3s * (1 + factors)))
                        mask_val = cv2.inRange(hsv_image[:, :, 2], int(q1v * (1 - factorv)), int(q3v * (1 + factorv)))
                        # mask_hsv = (mask_hue & mask_sat & mask_val) | skeleton_binary  # Take only concordances in the three channels
                        mask_hsv = (mask_sat & mask_val) | skeleton_binary  # Take only concordances in the three channels
                        mask_hsv = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB)  # Equivalent to cv2.merge((mask_hsv, mask_hsv, mask_hsv))
                        # Apply a closing operation and a dilation on the mask

                        mask_hsv |= cv2.cvtColor(skeleton_binary, cv2.COLOR_GRAY2RGB)

                        acc_ranges += (time.time() - ranges_0)

                        morp_0 = time.time()

                        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_closing, iterations=1)
                        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel_closing, iterations=1)
                        mask_hsv = cv2.dilate(mask_hsv, kernel_dilation2, iterations=1)
                        mask_hsv = mask_hsv & cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB)

                        acc_morp += (time.time() - morp_0)

                        error = False
                    except:
                        mask_hsv = np.zeros_like(ground_truth)
                        print("Error")
                        error = True

                end_t = time.time()


                print(f"Mediapipe ms: {acc_mp/iters*1000}")
                print(f"Morp ms: {acc_morp/iters*1000}")
                print(f"Cvt ms: {acc_cvt/iters*1000}")
                print(f"Ranges ms: {acc_ranges/iters*1000}")
                print(f"total ms: {(end_t-t0)/iters*1000}")

                if False:
                    ground_truth = ground_truth & cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB)
                    js = jaccard_score(ground_truth.flatten(), mask_hsv.flatten(), pos_label=255, zero_division=0)
                    if not error:
                        ious.append(js)
                        dir_ious.append(js)
                        image_ious.append((image_path, js))
                # print(f"IoU: {js}")

                if True:
                    cv2.imshow("MP", out)
                    # cv2.imshow("Stage 1", skeleton_binary)
                    cv2.imshow("Stage 2", mask_s2)

                    image2 = image & mask_hsv  # cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB) &
                    cv2.imshow("masked", image2)
                    cv2.imshow("mask a", mask_hsv)
                    colors_dict_temp = {"255": [0, 255, 0, 100]}
                    colors_dict_temp2 = {"255": [0, 0, 255, 100]}
                    masked_gt = draw_mask(image, cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB) & ground_truth, draw_zero=False, alpha=0.5, colors_dict=colors_dict_temp2)
                    cv2.imwrite("test_mask.png", masked_gt)
                    cv2.imshow("gt", masked_gt)

                    # colors_dict_temp = None
                    masked_output = draw_mask(image, mask_hsv, draw_zero=False, alpha=0.5, colors_dict=colors_dict_temp)
                    cv2.imwrite("test.png", masked_output)
                    cv2.imwrite("test_mp.png", out)
                    cv2.imshow("output", masked_output)
                    cv2.waitKey(0)

            print(f"FPS: {len([f for f in os.listdir(dir) if len(f) == 9])/(time.time() - t0)}")
            print(np.mean(dir_ious), np.std(dir_ious))
            print(sorted(image_ious[:5], key=lambda x: x[1], reverse=True))

        print(np.mean(ious), np.std(ious))
