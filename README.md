# In Browser StyleGAN Editing

## How it works
Models, weights, and architectures adopted from [encoder4editing](https://github.com/omertov/encoder4editing) are being converted to ONNX model format.

The models are hosted on [this](https://huggingface.co/jqhisme/e4e-ffhq-onnx/tree/main) huggingface repo. They can be fetched by
```javascript
fetch("https://huggingface.co/jqhisme/e4e-ffhq-onnx/tree/main/*")
```

In browser inference is ran by onnx web runtime, and FFQH style face alignment is done using ml5.js.

Visit live demo on github pages!