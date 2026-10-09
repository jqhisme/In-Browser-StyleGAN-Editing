from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
import onnxruntime as ort
import onnx

parser = argparse.ArgumentParser(description="Quantize the e4e encoder model.")
parser.add_argument('model')
args = parser.parse_args()

if args.model == 'encoder' or args.model == 'decoder':
    MODEL = Path(f"onnx-export/e4e_{args.model}.onnx")
else:
    raise ValueError(f"Invalid model argument: {args.model}")

OUTPUT = Path(f"onnx-export/e4e_{args.model}_int8.onnx")
CALIBRATION_PATH = Path("calibration_faces") 

class FaceReader(CalibrationDataReader):
    def __init__(self, folder: Path, encoder = None, get_latent = False):
        paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            raise ValueError(f"No calibration images found in {folder}")

        self.images = iter(paths)
        self.encoder = encoder
        self.get_latent = get_latent

    def get_next(self):
        path = next(self.images, None)
        if path is None:
            return None

        # Match the browser input: RGB, 256×256, NCHW, values in [-1, 1].
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (256, 256), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32)

        tensor = (pixels / 127.5 - 1.0).transpose(2, 0, 1)[None]
        if self.encoder is not None and self.get_latent:
            input_name = self.encoder.get_inputs()[0].name
            output_name = self.encoder.get_outputs()[0].name
            latent = self.encoder.run([output_name], {input_name: tensor})[0]
            return {"image": np.ascontiguousarray(latent)}
        return {"image": np.ascontiguousarray(tensor)}

if args.model == 'decoder':
    #encoder = onnx.load("onnx-export/e4e_encoder.onnx")
    provider = ['CUDAExecutionProvider','CPUExecutionProvider']
    session = ort.InferenceSession("onnx-export/e4e_encoder.onnx", providers=provider)
    quantize_args = {
        "encoder": session,
        "get_latent": True
    }
else:
    quantize_args = {}

quantize_static(
    model_input=MODEL,
    model_output=OUTPUT,
    calibration_data_reader=FaceReader(CALIBRATION_PATH, **quantize_args),
    calibrate_method=CalibrationMethod.MinMax,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    per_channel=True,
    use_external_data_format=True,
)