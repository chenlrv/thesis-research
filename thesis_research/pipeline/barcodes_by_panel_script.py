import requests

url = "https://github.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/raw/Main/_code/FOV%20QC/barcodes_by_panel.RDS"
out_path = "../../../resources/barcodes_by_panel.RDS"

r = requests.get(url)
r.raise_for_status()

with open(out_path, "wb") as f:
    f.write(r.content)

print("Downloaded bytes:", len(r.content))
