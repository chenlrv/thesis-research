import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg' — needs an interactive backend
import matplotlib.pyplot as plt
import tifffile
import numpy as np

step = 10  # adjust based on memory
he_l321 = r"D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\20251230_H&E_ReuvenStein\32_05.vsi.Collection\32_05_20x_BF_01.tif"
he_l34 = r"D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\20251230_H&E_ReuvenStein\32_06.vsi.Collection\32_06_20x_BF_02.tif"

with tifffile.TiffFile(he_l321) as tif:
    he_thumb = tif.pages[0].asarray()[::step, ::step]

# Transform thumbnail for display
# he_display = np.fliplr(he_thumb)
he_display = np.rot90(he_thumb, k=-1)
he_display = np.flipud(he_display)
# he_display = np.fliplr(he_display)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(he_display)
plt.title("Click to draw polygon. Right-click to close and finish.")

points = []  # will store coords in THUMBNAIL transformed space


# [(np.float64(14929.417403783922), np.float64(20067.292672319803)),
# (np.float64(13074.22470638442), np.float64(23356.56070635941)),
# (np.float64(12311.660836901192), np.float64(22889.917144436837))]
# cosmx:


points = []
first_click = True


def onclick(event):
    global first_click
    if event.xdata is None:
        return
    if event.button == 1:
        if first_click:
            first_click = False
            return  # ignore first click
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "r.", markersize=8)
        if len(points) > 1:
            ax.plot([points[-2][0], points[-1][0]], [points[-2][1], points[-1][1]], "r-")
        fig.canvas.draw()
    elif event.button == 3:
        if len(points) > 2:
            ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], "r-")
            fig.canvas.draw()
            plt.close()


fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
thumb_h, thumb_w = he_thumb.shape[:2]

points_full_res = [((thumb_w - x) * step, y * step) for x, y in points]
# Scale back up to full H&E pixel space
# points_full_res = [(x * step, y * step) for x, y in points]
print("Polygon in full H&E space:", points_full_res)

######

# adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\sample_L321_adata.h5ad")
#
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.scatter(adata.obs['CenterX_global_px'], adata.obs['CenterY_global_px'],
#            s=0.1, c='black', alpha=0.3)
#
# plt.title("Click the SAME 3 landmarks in the SAME ORDER as you did on the H&E")
#
# cell_pts = []
#
#
# def onclick(event):
#     if event.xdata is None:
#         return
#     if event.button == 1:
#         cell_pts.append((event.xdata, event.ydata))
#         ax.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=2)
#         ax.annotate(str(len(cell_pts)), (event.xdata, event.ydata),
#                     color='red', fontsize=12)
#         print(f"Point {len(cell_pts)}: ({event.xdata:.1f}, {event.ydata:.1f})")
#         fig.canvas.draw()
#
#         if len(cell_pts) == 3:
#             print("\nAll 3 points collected — you can close the window")
#
#
# fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()
#
# print("Cell space points:", cell_pts)
