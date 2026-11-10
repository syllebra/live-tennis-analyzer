import torch
from torchvision.models import mobilenet_v3_large, resnet50
from training.keypoint_training import KeypointModel

# Load your trained model
# backbone = mobilenet_v3_large(pretrained=True)
backbone = resnet50()
num_keypoints = 14
model = KeypointModel(backbone, num_keypoints).to('cuda')
model.load_state_dict(torch.load('models/keypoint_detector_resnet.pth'))
model.eval()

# Dummy input for ONNX export
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/keypoint_detector_resnet.onnx",
    input_names=["input"],
    output_names=["output"],
    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    # opset_version=11
)
