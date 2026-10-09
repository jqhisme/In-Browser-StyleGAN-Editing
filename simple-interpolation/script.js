// global variables
let faceMeshPromise;
let faceMeshDetectionQueue = Promise.resolve();
const uploadVersions = new Map();


async function init(){

    // set up file input and button
    let fileInput1 = document.getElementById("img-upload-1");
    let fileInput2 = document.getElementById("img-upload-2");
    fileInput1.addEventListener("change", (event)=>onImageUpload(event, "cnv-1"));
    fileInput2.addEventListener("change", (event)=>onImageUpload(event, "cnv-2"));


    // wait for model to load
    const models = await loadModel();
    document.getElementById("onnx-provider").textContent = "Prioritizing WebGPU and falling back to WASM(CPU)";
    document.getElementById("model-status").textContent = "Loaded";
    // let saveButton = document.createElement("button");
    // saveButton.textContent = "Save models to IndexDB";
    // document.getElementById("options").appendChild(saveButton);
    // saveButton.addEventListener("click", ()=>{
    //     saveModelsToIndexDB(models);
    // });

    if(!models) return;
    const { encoder, decoder } = models;

    let button   = document.getElementById("mix");
    // setup mixing logic
    button.addEventListener("click", ()=>{
        remix(encoder, decoder,10);
    });

}

async function onImageUpload(event,canvasID){
    const file = event.target.files[0];
    if (!file) return;

    const version = (uploadVersions.get(canvasID) ?? 0) + 1;
    uploadVersions.set(canvasID, version);
    const canvas = document.getElementById(canvasID);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    const url = URL.createObjectURL(file);
    try {
        const image = new Image();
        image.src = url;
        await image.decode();

        const faces = await detectUploadedFace(image);
        if (version !== uploadVersions.get(canvasID)) return;
        if (!faces.length) throw new Error("No face detected in the uploaded image.");

        alignFaceToCanvas(image, faces[0].keypoints, canvas);
    } catch (error) {
        console.error("Could not align uploaded face:", error);
        if (version === uploadVersions.get(canvasID)) alert(error.message);
    } finally {
        URL.revokeObjectURL(url);
        event.target.value = "";
    }
}

async function detectUploadedFace(image) {
    const mesh = await (faceMeshPromise ??= loadFaceMeshModel().catch(error => {
        faceMeshPromise = undefined;
        throw error;
    }));
    const detection = faceMeshDetectionQueue.then(async () => {
        // FaceMesh tracks the previous face region. A new upload needs a fresh search.
        mesh.model.reset();
        return mesh.detect(image);
    });
    faceMeshDetectionQueue = detection.catch(() => {});
    return detection;
}

function alignFaceToCanvas(image, keypoints, canvas) {
    // MediaPipe FaceMesh eye contours and outer mouth corners, in source pixels.
    const average = (indices) => ({
        x: indices.reduce((sum, i) => sum + keypoints[i].x, 0) / indices.length,
        y: indices.reduce((sum, i) => sum + keypoints[i].y, 0) / indices.length
    });
    const eyeLeft = average([33, 133, 159, 145]);
    const eyeRight = average([362, 263, 386, 374]);
    const mouthLeft = keypoints[61];
    const mouthRight = keypoints[291];
    const eyeAvg = { x: (eyeLeft.x + eyeRight.x) / 2, y: (eyeLeft.y + eyeRight.y) / 2 };
    const mouthAvg = { x: (mouthLeft.x + mouthRight.x) / 2, y: (mouthLeft.y + mouthRight.y) / 2 };
    const eyeToEye = { x: eyeRight.x - eyeLeft.x, y: eyeRight.y - eyeLeft.y };
    const eyeToMouth = { x: mouthAvg.x - eyeAvg.x, y: mouthAvg.y - eyeAvg.y };

    // FFHQ oriented-square geometry from alignment.py, using FaceMesh landmarks.
    const axis = { x: eyeToEye.x + eyeToMouth.y, y: eyeToEye.y - eyeToMouth.x };
    const axisLength = Math.hypot(axis.x, axis.y);
    if (!axisLength) throw new Error("Face landmarks could not define an alignment crop.");
    const radius = Math.max(Math.hypot(eyeToEye.x, eyeToEye.y) * 2,
                            Math.hypot(eyeToMouth.x, eyeToMouth.y) * 1.8);
    const x = { x: axis.x * radius / axisLength, y: axis.y * radius / axisLength };
    const y = { x: -x.y, y: x.x };
    const center = { x: eyeAvg.x + eyeToMouth.x * 0.1,
                     y: eyeAvg.y + eyeToMouth.y * 0.1 };

    const half = canvas.width / 2;
    const outputToSource = new DOMMatrix([
        x.x / half, x.y / half, y.x / half, y.y / half,
        center.x - x.x - y.x, center.y - x.y - y.y
    ]);
    const sourceToOutput = outputToSource.inverse();
    const context = canvas.getContext("2d");
    context.save();
    context.resetTransform();
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.setTransform(sourceToOutput.a, sourceToOutput.b,
                         sourceToOutput.c, sourceToOutput.d,
                         sourceToOutput.e, sourceToOutput.f);
    // Reflect the source at its edges when the rotated crop extends outside it.
    const corners = [
        { x: center.x - x.x - y.x, y: center.y - x.y - y.y },
        { x: center.x - x.x + y.x, y: center.y - x.y + y.y },
        { x: center.x + x.x + y.x, y: center.y + x.y + y.y },
        { x: center.x + x.x - y.x, y: center.y + x.y - y.y }
    ];
    const needsPadding = corners.some(point =>
        point.x < 0 || point.x > image.naturalWidth ||
        point.y < 0 || point.y > image.naturalHeight
    );
    if (needsPadding) {
        const xTiles = [[0, -1], [0, 1], [2 * image.naturalWidth, -1]];
        const yTiles = [[0, -1], [0, 1], [2 * image.naturalHeight, -1]];
        for (const [tx, sx] of xTiles) {
            for (const [ty, sy] of yTiles) {
                context.save();
                context.transform(sx, 0, 0, sy, tx, ty);
                context.drawImage(image, 0, 0);
                context.restore();
            }
        }
    } else {
        context.drawImage(image, 0, 0);
    }
    context.restore();
}

async function loadModel(){
    try{

        const base = "https://huggingface.co/jqhisme/e4e-ffhq-onnx/resolve/main/";

        const encoder = await ort.InferenceSession.create(
            base + "e4e_encoder_int8.onnx",
            {
                executionProviders: ["webgl","wasm"],
                externalData: [{
                    path: "e4e_encoder_int8.onnx.data",
                    data: base + "e4e_encoder_int8.onnx.data"
                }]
            }
        );

        const decoder = await ort.InferenceSession.create(
            base + "e4e_decoder_int8.onnx",
            {
                executionProviders: ["webgl","wasm"],
                externalData: [{
                    path: "e4e_decoder_int8.onnx.data",
                    data: base + "e4e_decoder_int8.onnx.data"
                }]
            }
        );

        console.log("Model loaded successfully");
        return { encoder, decoder };
    } catch (error) {
        console.error("Error loading model:", error);
    }
    return null;
}   


async function remix(encoder, decoder, steps){
    let cxt1 = document.getElementById("cnv-1").getContext("2d");
    let cxt2 = document.getElementById("cnv-2").getContext("2d");

    // load both images to tensors for the model
    const tensor1 = await ort.Tensor.fromImage(resizeCanvasForEncoder(cxt1.canvas),{
        tensorFormat: "RGB",
        tensorLayout: "NCHW"
    });
    const tensor2 = await ort.Tensor.fromImage(resizeCanvasForEncoder(cxt2.canvas),{
        tensorFormat: "RGB",
        tensorLayout: "NCHW"
    });
    // Match e4e's ToTensor() followed by Normalize(0.5, 0.5).
    for (const tensor of [tensor1, tensor2]) {
        for (let i = 0; i < tensor.data.length; i++) {
            tensor.data[i] = tensor.data[i] * 2 - 1;
        }
    }

    // pass then through the encoder
    let latent1 = await encoder.run({ image: tensor1 });
    let latent2 = await encoder.run({ image: tensor2 });
    console.log(encoder.outputNames)
    
    
    latent1 = latent1[encoder.outputNames[0]] //encoder.outputNames[0] is 'w_plus x'
    latent2 = latent2[encoder.outputNames[0]]

    for(let i =0;i< steps+1;i++){
        let alpha = i/steps;
        const values = new Float32Array(latent1.data.length);

        for (let j = 0; j < latent1.data.length; j++) {
            values[j] = alpha * latent1.data[j] + (1 - alpha) * latent2.data[j];
        }

        const latent = new ort.Tensor("float32", values, latent1.dims);
        console.log("Running interpolation step " + i + "/" + steps);

        const outputs = await decoder.run({ image: latent });
        const imageTensor = outputs[decoder.outputNames[0]];

        const [batch, channels, height, width] = imageTensor.dims;
        if (batch !== 1 || channels !== 3) {
            throw new Error(`Unexpected decoder shape: ${imageTensor.dims}`);
        }

        const plane = width * height;
        const rgba = new Uint8ClampedArray(plane * 4);
        const rgb = imageTensor.data;

        // scale the value to 0-128 and set the alpha channel to 255
        for (let p = 0; p < plane; p++) {
            rgba[4 * p]     = (rgb[p] + 1) * 127.5;             // R
            rgba[4 * p + 1] = (rgb[plane + p] + 1) * 127.5;     // G
            rgba[4 * p + 2] = (rgb[2 * plane + p] + 1) * 127.5; // B
            rgba[4 * p + 3] = 255;                              // A
        }

        // putImageData writes pixels at native size; drawImage scales them.
        const outputCanvas = document.getElementById("output-cnv");
        const frameCanvas = document.createElement("canvas");
        frameCanvas.width = width;
        frameCanvas.height = height;
        frameCanvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);

        const outputContext = outputCanvas.getContext("2d");
        outputContext.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputContext.drawImage(frameCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
        await new Promise(resolve =>
            requestAnimationFrame(() => setTimeout(resolve, 0))
        );
    }
}

function resizeCanvasForEncoder(sourceCanvas) {
    const resized = document.createElement("canvas");
    resized.width = 256;
    resized.height = 256;
    const context = resized.getContext("2d", { willReadFrequently: true });
    context.drawImage(sourceCanvas, 0, 0, resized.width, resized.height);
    return context.getImageData(0, 0, resized.width, resized.height);
}

async function loadFaceMeshModel(){
    let ml5FaceMesh = await ml5.faceMesh({
        maxFaces:1,
        refineLandmarks: false,
        flipHorizontal: false
    });
    return ml5FaceMesh;
}

init();
