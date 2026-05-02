from pathlib import Path
import onnx
import torch
from torch import nn
import sys
sys.path.insert(0, "/Users/minseong/my/opensource/project")
from src.model import SegModel

DATA_FIELDS = ("raw_data", "float_data", "int32_data", "int64_data")

def export_structure_only_onnx(
    model: nn.Module,
    input_size: list,
    output_path: Path,
    input_names: list = ["input"],
    output_names: list = ["output"],
    opset: int = 18,
) -> Path:
    dummy_input = torch.randn(*input_size)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            external_data=False,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
        )
    onnx_model = onnx.load(output_path)
    for initializer in onnx_model.graph.initializer:
        for field in DATA_FIELDS:
            initializer.ClearField(field)
    onnx.save(onnx_model, output_path)
    return output_path

model = SegModel(num_classes=21, pretrained=False, aspp_ch=128)
model.eval()

out = export_structure_only_onnx(
    model=model,
    input_size=[1, 3, 480, 640],
    output_path=Path("/Users/minseong/my/opensource/project/model_structure.onnx"),
)
print(f"Exported: {out}  ({out.stat().st_size / 1024:.1f} KB)")