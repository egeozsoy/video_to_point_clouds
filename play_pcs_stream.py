import open3d as o3d
from pathlib import Path
from time import time

vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)
geometry = None
all_pcds = sorted(list(Path('pcs').glob('*.pcd')))
for pcd_path in all_pcds:
    pc = o3d.io.read_point_cloud(str(pcd_path))
    start = time()
    freq = 0.001
    while True:
        if time() - start > freq:
            if geometry is None:
                geometry = pc
                vis.add_geometry(geometry)
            else:
                geometry.points = pc.points
                geometry.colors = pc.colors
                vis.update_geometry(geometry)
            break
        vis.poll_events()
        vis.update_renderer()
