from roboflow import Roboflow

# rf = Roboflow(api_key="hRrqiECkoEkefllT6CWH")
# project = rf.workspace("small-object-detections-smart-surveillance-system").project(
#     "object-detection-axukj-li4tq"
# )
# version = project.version(1)
# dataset = version.download("coco")


# from roboflow import Roboflow

# rf = Roboflow(api_key="apUiV7snDplgUzUnTj5Y")
# project = rf.workspace("small-object-detections-smart-surveillance-system").project(
#     "ssd-object-detection-b38kl"
# )
# version = project.version(6)
# dataset = version.download("coco")


from roboflow import Roboflow

rf = Roboflow(api_key="apUiV7snDplgUzUnTj5Y")
project = rf.workspace("small-object-detections-smart-surveillance-system").project(
    "ssd-object-detection-b38kl"
)
version = project.version(7)
dataset = version.download("coco")
