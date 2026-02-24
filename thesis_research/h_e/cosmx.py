import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anndata as ad

adata = ad.read_h5ad(r'D:\thesis-research\resources\cache\sample_L321_adata.h5ad')
# cells = pd.read_csv("cells.csv")


fig, ax = plt.subplots(figsize=(12, 20))
ax.set_aspect('equal')  # force equal scaling on both axes

ax.scatter(adata.obs['CenterX_global_px'], adata.obs['CenterY_global_px'],
           s=0.1, c='black', alpha=0.1)
plt.title("Click the SAME 3 landmarks in the SAME ORDER as H&E. Right-click to finish.")

# Your H&E landmarks for reference
he_landmarks = [
(np.float64(14854.983105877585), np.float64(16475.927063698906)),
(np.float64(13016.421351842022), np.float64(13224.818452997064)),
(np.float64(12165.095270029975), np.float64(14148.597818367583))

]

#[(np.float64(89796.13197173372), np.float64(67251.54069504245)]
#(np.float64(98750.41282708588), np.float64(51910.65522663074)), (np.float64(101743.81746169561), np.float64(54761.51678340192))
print("Click CosMx equivalents of these H&E points (in order):")
for i, pt in enumerate(he_landmarks):
    print(f"  Point {i+1}: {pt}")

cosmx_pts = []

def onclick(event):
    if event.xdata is None:
        return
    if event.button == 1:
        cosmx_pts.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=2)
        ax.annotate(str(len(cosmx_pts)), (event.xdata, event.ydata),
                    color='red', fontsize=12)
        print(f"Point {len(cosmx_pts)}: ({event.xdata:.1f}, {event.ydata:.1f})")
        fig.canvas.draw()

        if len(cosmx_pts) == 5:
            print("\nAll 3 points collected!")
            print("CosMx landmarks:", cosmx_pts)
            plt.close()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

cosmx_landmarks = cosmx_pts