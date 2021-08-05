import skvideo

skvideo.setFFmpegPath('/usr/local/Cellar/ffmpeg/4.4_2/bin')
import skvideo.io
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import open3d as o3d

IMAGE_SIZE = (1280, 720)
DEPTH_TRUNC = 0.5
FOCAL_LENGTH = 2772  # width / (np.tan(np.deg2rad(26mm) / 2) * 2),  # not sure if we should multiply with 2, also this value depends on pixel size
EVERY_N_FRAME = 1

def prep_model():
    model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform, device


def pred_inverse_depth(model, transform, rgb, device):
    input_batch = transform(rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().clamp(min=0).cpu().numpy()
        # plt.imshow(output)
        # plt.show()
        return prediction


def compute_pc_from_rgb_depth(img, depth_pred):
    rgb = o3d.geometry.Image(img)
    depth = o3d.geometry.Image((1 / depth_pred))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
    width, height = img.shape[1], img.shape[0]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width,
                                                   height,
                                                   2772,
                                                   2772,
                                                   width / 2,
                                                   height / 2)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Flip it, otherwise the pointcloud will be upside down
    pc.transform([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([pc])

    return pc


def main():
    possible_names = ['input.MOV', 'input.mov', 'input.mp4', 'input.MP4']
    output_folder = Path('pcs')
    if not output_folder.exists():
        output_folder.mkdir()

    video_path = None
    for name in possible_names:
        if Path(name).exists():
            video_path = Path(name)
            break

    if video_path is None:
        print('No video found')
        return

    model, transform, device = prep_model()

    for idx, rgb in enumerate(skvideo.io.vreader(str(video_path))):
        if idx % EVERY_N_FRAME != 0:
            continue

        rgb = np.asarray(Image.fromarray(rgb).resize(IMAGE_SIZE))
        depth_pred = pred_inverse_depth(model, transform, rgb, device)
        pc = compute_pc_from_rgb_depth(rgb, depth_pred)
        out_str = str(idx).zfill(6)
        o3d.io.write_point_cloud(str(output_folder / f'{out_str}.ply'), pc)


if __name__ == '__main__':
    main()
