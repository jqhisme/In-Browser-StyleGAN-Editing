// global variables
let faceMeshPromise;
let video;
let cameraCanvas;
let cameraCtx;
let detectionCanvas;
let detectionCtx;
let drawCanvas;
let drawCtx;
let faceMeshModel;
let encoder;
let decoder;
const latents = [null, null];
const alignedUploads = [false, false];
const uploadVersions = [0, 0];
let detectionQueue = Promise.resolve();
let lastDetectionSource = null;
let encoderQueue = Promise.resolve();
let latestNose = null;
let lastRenderedAlpha = null;
let mixInFlight = false;
let emptyCameraDetections = 0;
let latentDirections;
let activeLatentDirection;

async function init(){

    document.getElementById("img-upload1").addEventListener("change", event => onImageUpload(event, 0));
    document.getElementById("img-upload2").addEventListener("change", event => onImageUpload(event, 1));

    cameraCanvas = document.getElementById("cnv-camera");
    cameraCtx = cameraCanvas.getContext("2d");
    drawCanvas = document.getElementById("cnv-draw");
    drawCtx = drawCanvas.getContext("2d");
    video = document.createElement("video");
    video.hidden = true;
    await startCamera();

    getFaceMeshModel().then(() => detectCameraFrames()).catch(error => {
        console.error("Could not start face tracking:", error);
        setMixStatus("Face tracking failed.");
    });
    latentDirections = await loadLatentDirections();
    const models = await loadModel();
    if (!models) {
        document.getElementById("model-status").textContent = "Failed to load";
        return;
    }
    ({ encoder, decoder } = models);
    document.getElementById("onnx-provider").textContent = "Prioritizing WebGPU and falling back to WASM (CPU)";
    document.getElementById("model-status").textContent = "Loaded";
    setMixStatus(alignedUploads.some(Boolean)
        ? "Encoding uploaded face..."
        : "Upload two face images to begin mixing.");

    for (let index = 0; index < alignedUploads.length; index++) {
        if (alignedUploads[index] && !latents[index]) {
            encodeUploadedFace(index).catch(error => console.error("Could not encode uploaded face:", error));
        }
    }

}

function setMixStatus(message) {
    document.getElementById("mix-status").textContent = message;
}

function getFaceMeshModel() {
    return faceMeshPromise ??= loadFaceMeshModel().then(model => {
        faceMeshModel = model;
        return model;
    }).catch(error => {
        faceMeshPromise = undefined;
        throw error;
    });
}

async function detectFaces(source) {
    const model = faceMeshModel ?? await getFaceMeshModel();
    const result = detectionQueue.then(async () => {
        // The TF.js FaceMesh detector tracks the previous face region. An
        // uploaded photo is a new scene, so discard that region on source changes.
        if (lastDetectionSource !== source) {
            model.model.reset();
            lastDetectionSource = source;
        }
        try {
            return await model.detect(source);
        } catch (error) {
            lastDetectionSource = null;
            throw error;
        }
    });
    detectionQueue = result.catch(() => {});
    return result;
}

async function onImageUpload(event, index){
    const file = event.target.files[0];
    if (!file) return;

    const version = ++uploadVersions[index];
    const canvas = document.getElementById(`cnv-uploaded${index + 1}`);
    alignedUploads[index] = false;
    latents[index] = null;
    lastRenderedAlpha = null;
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    const url = URL.createObjectURL(file);
    try {
        const image = new Image();
        image.src = url;
        await image.decode();
        console.info("Detecting the original uploaded image", {
            width: image.naturalWidth,
            height: image.naturalHeight
        });

        const faces = await detectFaces(image);
        if (version !== uploadVersions[index]) return;
        if (!faces.length) throw new Error("No face detected in the uploaded image.");

        alignFaceToCanvas(image, faces[0].keypoints, canvas);
        alignedUploads[index] = true;
        setMixStatus(encoder ? "Encoding uploaded face..." : "Face aligned; waiting for the image model.");
        if (encoder) await encodeUploadedFace(index);
    } catch (error) {
        console.error("Could not align uploaded face:", error);
        if (version === uploadVersions[index]) setMixStatus(error.message);
    } finally {
        URL.revokeObjectURL(url);
    }
}

function encodeUploadedFace(index) {
    const version = uploadVersions[index];
    const canvas = document.getElementById(`cnv-uploaded${index + 1}`);
    const task = encoderQueue.then(async () => {
        if (version !== uploadVersions[index] || !alignedUploads[index]) return;
        if (canvas.width !== 256 || canvas.height !== 256) {
            throw new Error("The aligned face canvas must be 256×256 pixels.");
        }
        const alignedPixels = canvas.getContext("2d").getImageData(0, 0, 256, 256);
        const tensor = await ort.Tensor.fromImage(alignedPixels, {
            tensorFormat: "RGB",
            tensorLayout: "NCHW"
        });
        // Match e4e's ToTensor() and Normalize(0.5, 0.5).
        for (let i = 0; i < tensor.data.length; i++) tensor.data[i] = tensor.data[i] * 2 - 1;
        const outputs = await encoder.run({ image: tensor });
        if (version !== uploadVersions[index]) return;
        latents[index] = outputs[encoder.outputNames[0]];
        lastRenderedAlpha = null;
        setMixStatus(latents.every(Boolean) ? "Move your nose left and right to mix." : "Upload the other face image.");
    });
    encoderQueue = task.catch(() => {});
    return task;
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

async function loadLatentDirections(){
    const base = "https://huggingface.co/jqhisme/e4e-ffhq-onnx/resolve/main/";
    const latentDirections = await fetch(base + "latent_directions.json").then(res => res.json());
    // Copy the 512-value direction into each of the 18 StyleGAN layers.
    for (const key of Object.keys(latentDirections)) {
        const direction = latentDirections[key];
        if (Array.isArray(direction) && direction.length === 512) {
            const repeated = new Float32Array(18 * 512);
            for (let i = 0; i < 18; i++) {
                repeated.set(direction, i * 512);
            }
            latentDirections[key] = new ort.Tensor("float32", repeated, [1, 18, 512]);
            activeLatentDirection ??= latentDirections[key];
        }
    }
    if (!activeLatentDirection) throw new Error("No 512-value latent direction was found.");
    return latentDirections;
}
async function loadModel(){
    try{

        const base = "https://huggingface.co/jqhisme/e4e-ffhq-onnx/resolve/main/";

        const encoder = await ort.InferenceSession.create(
            base + "e4e_encoder_int8.onnx",
            {
                executionProviders: ["webgpu","wasm"],
                externalData: [{
                    path: "e4e_encoder_int8.onnx.data",
                    data: base + "e4e_encoder_int8.onnx.data"
                }]
            }
        );

        const decoder = await ort.InferenceSession.create(
            base + "e4e_decoder.onnx",
            {
                executionProviders: ["webgpu","wasm"],
                externalData: [{
                    path: "e4e_decoder.onnx.data",
                    data: base + "e4e_decoder.onnx.data"
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

async function detectCameraFrames() {
    try {
        // Detect the same pixels shown in the camera preview. A canvas also
        // avoids differences in how GPU backends read a live video element.
        detectionCtx.drawImage(video, 0, 0, detectionCanvas.width, detectionCanvas.height);
        const faces = await detectFaces(detectionCanvas);
        if (!Array.isArray(faces)) {
            throw new TypeError(`FaceMesh returned ${faces} instead of an array.`);
        }
        if (faces.length === 0) {
            emptyCameraDetections++;
            if (emptyCameraDetections === 30) {
                console.warn("FaceMesh found no face in 30 camera frames", {
                    videoWidth: video.videoWidth,
                    videoHeight: video.videoHeight,
                    readyState: video.readyState,
                    currentTime: video.currentTime
                });
                setMixStatus("Camera is live, but no face is detected. Face the camera in good light.");
            }
        } else {
            if (emptyCameraDetections >= 30 && latents.every(Boolean)) {
                setMixStatus("Move your nose left and right to mix.");
            }
            emptyCameraDetections = 0;
        }
        latestNose = faces[0]?.keypoints?.[1] ?? null;
        if (latestNose && latents.every(Boolean)) requestMix();
        requestAnimationFrame(detectCameraFrames);
    } catch (error) {
        console.error("Face tracking stopped:", error);
        setMixStatus("Face tracking stopped. Check the console.");
    }
}

function requestMix() {
    if (mixInFlight || !latestNose || !decoder || !latents.every(Boolean)) return;
    const alphaX = Math.max(0, Math.min(1, latestNose.x / video.videoWidth));
    const alphaY = Math.max(0, Math.min(1, latestNose.y / video.videoHeight));
    if (lastRenderedAlpha &&
        Math.abs(alphaX - lastRenderedAlpha.x) < 0.01 &&
        Math.abs(alphaY - lastRenderedAlpha.y) < 0.01) return;

    const inputs = [...latents];
    mixInFlight = true;
    remix(inputs[0], inputs[1], alphaX, alphaY).then(frame => {
        if (inputs[0] !== latents[0] || inputs[1] !== latents[1]) return;
        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        drawCtx.drawImage(frame, 0, 0, drawCanvas.width, drawCanvas.height);
        lastRenderedAlpha = { x: alphaX, y: alphaY };
    }).catch(error => {
        console.error("Could not decode mix:", error);
        setMixStatus("Mixing failed. Check the console.");
    }).finally(() => {
        mixInFlight = false;
    });
}

async function startCamera(){
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.playsInline = true;
    await video.play();
    cameraCanvas.width = video.videoWidth;
    cameraCanvas.height = video.videoHeight;
    detectionCanvas = document.createElement("canvas");
    detectionCanvas.width = video.videoWidth;
    detectionCanvas.height = video.videoHeight;
    detectionCtx = detectionCanvas.getContext("2d");
    const drawCameraFrame = () => {
        cameraCtx.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);
        if (latestNose) {
            cameraCtx.beginPath();
            cameraCtx.arc(latestNose.x, latestNose.y, 5, 0, 2 * Math.PI);
            cameraCtx.fillStyle = "green";
            cameraCtx.fill();
        }
        requestAnimationFrame(drawCameraFrame);
    };
    drawCameraFrame();
}

async function remix(latent1, latent2, alphaX, alphaY){
    const values = new Float32Array(latent1.data.length);
    alphaY = (alphaY-0.5) * 10; // range from -5 to 5, according to experiments in e4e

    for (let j = 0; j < latent1.data.length; j++) {
        values[j] = alphaX * latent1.data[j] + (1 - alphaX) * latent2.data[j];
        values[j] += alphaY * activeLatentDirection.data[j];
    }

    const latent = new ort.Tensor("float32", values, latent1.dims);
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
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = width;
    frameCanvas.height = height;
    frameCanvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);

    return frameCanvas;
    
}

async function loadFaceMeshModel(){
    let ml5FaceMesh = await ml5.faceMesh({
        maxFaces:1,
        refineLandmarks: false,
        flipHorizontal: false
    });
    return ml5FaceMesh;
}

init().catch(error => {
    console.error("Could not initialize camera mixing:", error);
    setMixStatus(`Could not initialize: ${error.message}`);
});
