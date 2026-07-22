import os

base_dir = r"c:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\webapp\training\phase8"
src = os.path.join(base_dir, "train_ssvb_production.py")
dst = os.path.join(base_dir, "train_convmoe_production.py")

with open(src, "r") as f:
    content = f.read()

replacements = [
    ("from backend.runtime.ssvb_casa_ais import SSVBCASA_AIS", "from backend.runtime.conv_moe_mf import ConvMoE_MF"),
    ("'ssvb_casa_ais_production'", "'convmoe_mf_production'"),
    ("SSVBCASA_AIS", "ConvMoE_MF"),
    ("'batch_size':         256", "'batch_size':         64"),
    ("'ssl_epochs':         4", "'ssl_epochs':         0"),
    ("'ft_epochs':          8", "'ft_epochs':          50"),
    ("'lr_ft':              5e-4", "'lr_ft':              1e-3"),
    ("'lambda_conf':        0.15", "'lambda_conf':        0.10"),
    ("SSVB-CASA-AIS", "ConvMoE-MF")
]

for old, new in replacements:
    content = content.replace(old, new)

with open(dst, "w") as f:
    f.write(content)

print(f"Created {dst} successfully.")
