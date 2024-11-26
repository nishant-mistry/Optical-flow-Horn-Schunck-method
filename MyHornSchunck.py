
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch


def forward_accumulation(flows: list[torch.Tensor]) -> torch.Tensor:
    # flows: list of flow tensors (2, h, w)
    # return: cummulative flow tensor (2, h, w)
    cummualtive_flow = torch.zeros_like(flows[0])
    for flow in flows:
        h, w = cummualtive_flow.shape[-2:]
        coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h), indexing="xy"
            ),
            dim=-1,
        )
        flow_map = coordinates + flow.permute(1, 2, 0)
        flow_map[..., 0] = flow_map[..., 0] / (w - 1) * 2 - 1
        flow_map[..., 1] = flow_map[..., 1] / (h - 1) * 2 - 1
        new_flow = torch.nn.functional.grid_sample(
            flow.unsqueeze(0),
            flow_map.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        cummualtive_flow += new_flow.squeeze()

    return cummualtive_flow


def backward_accumulation(flows: list[torch.Tensor]) -> torch.Tensor:
    # flows: list of flow tensors (2, h, w)
    # return: cummulative flow tensor (2, h, w)
    cummulative_flow = flows[-1].clone()
    for flow in reversed(flows[:-1]):
        h, w = cummulative_flow.shape[-2:]
        coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h), indexing="xy"
            ),
            dim=-1,
        )
        flow_map = coordinates + flow.permute(1, 2, 0)
        flow_map[..., 0] = flow_map[..., 0] / (w - 1) * 2 - 1
        flow_map[..., 1] = flow_map[..., 1] / (h - 1) * 2 - 1
        new_flow = torch.nn.functional.grid_sample(
            cummulative_flow.unsqueeze(0),
            flow_map.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        cummulative_flow += new_flow.squeeze()
    return cummulative_flow


def accumulate_flows(flows: list[torch.Tensor], backward: bool = False) -> torch.Tensor:
    # flows: list of flow tensors (2, h, w)
    # backward: bool, if True, accumulate flows backward
    # return: cummulative flow tensor (2, h, w)
    if backward:
        return backward_accumulation(flows)
    else:
        return forward_accumulation(flows)


def draw_flow_hsv(flow: np.ndarray):
    hsv = np.zeros(flow.shape[:-1] + (3,), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def colour_wheel(samples=1024, clip_circle=True):

    xx, yy = np.meshgrid(np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))
    mag, _ = cv2.cartToPolar(xx, yy)

    flow = np.stack([xx, yy], axis=-1)
    flow_img = draw_flow_hsv(flow)

    if clip_circle == True:
        alpha = np.where(mag[..., None] > 1, 0, 255)
        flow_img[mag > 1, :] = 0
    else:
        alpha = np.ones(mag[..., None].shape) * 255

    return np.concatenate([flow_img, alpha], axis=-1)


def warp_image(image: np.ndarray, flow: np.ndarray):
    h, w = image.shape[:2]
    flow_map = np.array(
        [[np.array([x, y]) + flow[y, x] for x in range(w)] for y in range(h)],
        dtype=np.float32,
    )
    warped_image = cv2.remap(
        image, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR
    )
    return warped_image
"""
see readme for running instructions
"""

def plot_images(images, titles=None, cols=2, cmap="gray", size=10):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            ax.axis("off")
            if titles is not None:
                ax.set_title(titles[i])
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 1
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg



def draw_quiver(u,v,beforeImg):
    scale = 1
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()



#compute derivatives of the image intensity values along the x, y, time
def get_derivatives(img1, img2):
    #derivative masks
    eps = 0.23585 # video has 4.24 px/um scale
    delta_time = 0.1 # sec i.e. 10 fps
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 1.0 / (2*eps)
    y_kernel = np.array([[-1, -1], [1, 1]]) * 1.0 / (2*eps)
    t_kernel = np.ones((2, 2)) * 0.25 * 1 / delta_time

    fx = (convolve(img1,x_kernel) + convolve(img2,x_kernel))*0.5
    fy = (convolve(img1, y_kernel) + convolve(img2, y_kernel))*0.5
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)
    print("fx shape =",fx.shape)
    print("fy shape =",fy.shape)
    print("ft shape =",ft.shape)
    return [fx,fy, ft]



#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver
def computeHS(name1, name2, alpha, delta):

    # path = os.path.join(os.path.dirname(__file__), 'test images')
    # beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE)
    # afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE)

    # if beforeImg is None:
    #     raise NameError("Can't find image: \"" + name1 + '\"')
    # elif afterImg is None:
    #     raise NameError("Can't find image: \"" + name2 + '\"')

    # beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
    # afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)

    beforeImg= cv2.cvtColor(name1, cv2.COLOR_BGR2GRAY)
    afterImg= cv2.cvtColor(name2, cv2.COLOR_BGR2GRAY) 
    #removing noise
    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = convolve(u, avg_kernel)
        v_avg = convolve(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        # d = 4 * alpha**2 + fx**2 + fy**2
        d = alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    # draw_quiver(u, v, beforeImg)

    return [u, v]



if __name__ == '__main__':
    # parser = ArgumentParser(description = 'Horn Schunck program')
    # parser.add_argument('img1', type = str, help = 'First image name (include format)')
    # parser.add_argument('img2', type = str, help='Second image name (include format)')
    # args = parser.parse_args()
    videopath="/Users/Nishant/Downloads/20240208_Polished_Sample3_1015mmdisp-00000003.Avi"
    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frames per second =',fps, total_frames)
    ret, first_image = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        print("first image", first_image.shape)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames*0.5)-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, last_image = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        print("last image", last_image.shape) 
    
    # cv2.imshow('Original Image', first_image)
    # cv2.waitKey(0)
    # cv2.imshow('Final Image', last_image)
    # cv2.waitKey(0)

    # beforeImg= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    # afterImg= cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
    u,v = computeHS(first_image, last_image, alpha = 15, delta = 10**-3)
    flow = np.stack((u,v), axis=2, dtype=float)
    print(flow.shape)
    print(type(flow))
    # beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    # print(beforeImg.shape)
    # afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)
    # print(afterImg.shape)
    # cv2.imshow('Original Image Gray', beforeImg)
    # cv2.waitKey(0)
    # cv2.imshow('Final Image Gray', afterImg)
    # cv2.waitKey(0)

    flow_img = draw_flow_hsv(flow)
    warped_image = warp_image(first_image, flow)
    difference = warped_image - last_image

    optical_flow_color_wheel = colour_wheel()

    plot_images([first_image, last_image, flow_img, optical_flow_color_wheel, warped_image, difference], ['Source', 'Destination', 'Optical Flow', "Color_wheel", 'Warped Image', 'difference'])


    cap.release()
    cv2.destroyAllWindows()





