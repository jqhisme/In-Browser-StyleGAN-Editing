const MODEL_BASE = "https://huggingface.co/jqhisme/e4e-ffhq-onnx/resolve/main/";
const $ = id => document.getElementById(id);
const axisNames = ["x", "y", "z"];
const position = [0, 0, 0];
const velocity = [0, 0, 0];
const gravity = [0, 0, 0];
let encoder, decoder, directions, baseLatent, faceMeshPromise;
let uploadVersion = 0;
let renderVersion = 0;
let rendering = false;
let renderTimer;
let lastMotionTime = 0;
let motionSeen = false;
let motionTimeout;

function setStatus(kind, label, state = "") {
  $(`${kind}-status`).textContent = label;
  $(`${kind}-dot`).className = `status-dot ${state}`;
}

function message(text) { $("message").textContent = text; }
function clamp(value, lo, hi) { return Math.max(lo, Math.min(hi, value)); }

function resetMotion() {
  position.fill(0);
  velocity.fill(0);
  axisNames.forEach((name, index) => $(`value-${name}`).textContent = position[index].toFixed(2));
  queueRender();
}

function onMotion(event) {
  let sample = event.acceleration;
  if (!sample || ![sample.x, sample.y, sample.z].every(Number.isFinite)) {
    const total = event.accelerationIncludingGravity;
    if (!total || ![total.x, total.y, total.z].every(Number.isFinite)) return;
    const values = [total.x, total.y, total.z];
    values.forEach((value, i) => { gravity[i] += (value - gravity[i]) * 0.08; });
    sample = { x: values[0] - gravity[0], y: values[1] - gravity[1], z: values[2] - gravity[2] };
  }

  if (!motionSeen) {
    motionSeen = true;
    clearTimeout(motionTimeout);
    setStatus("motion", "Receiving motion", "ready");
    $("enable-motion").textContent = "Motion enabled";
    $("enable-motion").disabled = true;
    $("recenter").disabled = false;
    message(baseLatent ? "Move your phone to edit the portrait." : "Motion is ready. Choose a face photo.");
  }

  const now = performance.now();
  const dt = lastMotionTime ? clamp((now - lastMotionTime) / 1000, 0.01, 0.08) : 0.02;
  lastMotionTime = now;
  const components = [sample.x, sample.y, sample.z];
  let changed = false;
  components.forEach((raw, i) => {
    // DeviceMotion axes are fixed to portrait: right, top, out of screen.
    // Dead zone and damping make small hand tremors less likely to drift.
    const acceleration = Math.sign(raw) * Math.max(0, Math.abs(raw) - 0.35);
    velocity[i] = clamp((velocity[i] + acceleration * dt) * Math.exp(-2.4 * dt), -2, 2);
    const next = clamp(position[i] + velocity[i] * dt * 5, -1, 1);
    if (Math.abs(next - position[i]) > 0.003) changed = true;
    position[i] = next;
    $(`value-${axisNames[i]}`).textContent = next.toFixed(2);
  });
  if (changed) queueRender();
}

async function enableMotion() {
  if (!window.isSecureContext) {
    setStatus("motion", "HTTPS required", "error");
    message("Open this page over HTTPS to use the motion sensor.");
    return;
  }
  if (!("DeviceMotionEvent" in window)) {
    setStatus("motion", "Unavailable", "error");
    message("This browser does not expose a motion sensor.");
    return;
  }
  try {
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== "granted") {
        setStatus("motion", "Access denied", "error");
        message("Motion access was denied. You can try enabling it again in browser settings.");
        return;
      }
    }
    setStatus("motion", "Waiting for sensor…");
    message("Move the phone slightly to check the sensor.");
    window.addEventListener("devicemotion", onMotion);
    clearTimeout(motionTimeout);
    motionTimeout = setTimeout(() => {
      if (!motionSeen) {
        setStatus("motion", "No sensor data", "error");
        message("Permission may be enabled, but this browser has not sent motion data.");
      }
    }, 3500);
  } catch (error) {
    setStatus("motion", "Access unavailable", "error");
    message(`Could not enable motion: ${error.message}`);
  }
}

async function loadModels() {
  try {
    setStatus("model", "Loading directions…");
    const response = await fetch(MODEL_BASE + "latent_directions.json");
    if (!response.ok) throw new Error(`Directions request failed (${response.status})`);
    const json = await response.json();
    directions = Object.fromEntries(Object.entries(json).map(([name, value]) => [name, Array.isArray(value) ? value.flat(Infinity) : []])
      .filter(([, values]) => [512, 18 * 512].includes(values.length) && values.every(Number.isFinite)));
    const names = Object.keys(directions);
    if (!names.length) throw new Error("No usable directions found in latent_directions.json");
    axisNames.forEach((axis, i) => {
      const select = $(`axis-${axis}`);
      for (const name of names) select.add(new Option(name.replaceAll("_", " "), name));
      select.value = names[i % names.length];
    });

    const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    setStatus("model", "Loading encoder…");
    encoder = await ort.InferenceSession.create(MODEL_BASE + "e4e_encoder.onnx", {
      executionProviders: providers,
      externalData: [{ path: "e4e_encoder.onnx.data", data: MODEL_BASE + "e4e_encoder.onnx.data" }]
    });
    setStatus("model", "Loading decoder…");
    decoder = await ort.InferenceSession.create(MODEL_BASE + "e4e_decoder.onnx", {
      executionProviders: providers,
      externalData: [{ path: "e4e_decoder.onnx.data", data: MODEL_BASE + "e4e_decoder.onnx.data" }]
    });
    setStatus("model", "Ready", "ready");
    message("Models ready. Choose a face photo to begin.");
  } catch (error) {
    console.error("Model loading failed:", error);
    setStatus("model", "Failed to load", "error");
    message(`Could not load models: ${error.message}`);
  }
}

function queueRender() {
  if (!baseLatent || !decoder) return;
  renderVersion++;
  if (!rendering && !renderTimer) renderTimer = setTimeout(renderPortrait, 180);
}

async function renderPortrait() {
  renderTimer = undefined;
  if (!baseLatent || !decoder || rendering) return;
  rendering = true;
  const version = renderVersion;
  const latent = baseLatent;
  try {
    const values = new Float32Array(latent.length);
    values.set(latent);
    const strength = Number($("strength").value);
    axisNames.forEach((axis, i) => {
      const direction = directions[$(`axis-${axis}`).value];
      if (!direction) return;
      const offset = position[i] * strength;
      for (let j = 0; j < values.length; j++) values[j] += offset * direction[j % direction.length];
    });
    const output = await decoder.run({ image: new ort.Tensor("float32", values, [1, 18, 512]) });
    if (version !== renderVersion || latent !== baseLatent) return;
    drawOutput(output[decoder.outputNames[0]]);
    $("empty-state").hidden = true;
    $("busy-state").hidden = true;
  } catch (error) {
    console.error("Decoder failed:", error);
    $("busy-state").hidden = true;
    message(`Could not render this edit: ${error.message}`);
  } finally {
    rendering = false;
    if (version !== renderVersion) queueRender();
  }
}

function drawOutput(tensor) {
  const [batch, channels, height, width] = tensor.dims;
  if (batch !== 1 || channels !== 3) throw new Error(`Unexpected decoder output: ${tensor.dims}`);
  const plane = width * height;
  const rgba = new Uint8ClampedArray(plane * 4);
  const rgb = tensor.data;
  for (let i = 0; i < plane; i++) {
    rgba[4 * i] = (rgb[i] + 1) * 127.5;
    rgba[4 * i + 1] = (rgb[plane + i] + 1) * 127.5;
    rgba[4 * i + 2] = (rgb[2 * plane + i] + 1) * 127.5;
    rgba[4 * i + 3] = 255;
  }
  const canvas = $("output-cnv");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
}

async function uploadFace(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const version = ++uploadVersion;
  renderVersion++;
  baseLatent = null;
  resetMotion();
  $("busy-state").textContent = "Preparing face…";
  $("busy-state").hidden = false;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const mesh = await (faceMeshPromise ??= Promise.resolve().then(() => ml5.faceMesh({
      maxFaces: 1, refineLandmarks: false, flipHorizontal: false
    })).catch(error => {
      faceMeshPromise = undefined;
      throw error;
    }));
    const faces = await mesh.detect(image);
    if (!faces.length) throw new Error("No face was found. Try a clear, front-facing photo.");
    alignFaceToCanvas(image, faces[0].keypoints, $("aligned-cnv"));
    if (version !== uploadVersion) return;
    if (!encoder || !decoder) {
      $("busy-state").textContent = "Waiting for models…";
      await modelsReady;
    }
    if (!encoder || !decoder) throw new Error("The models could not load. Check the model status above.");
    $("busy-state").textContent = "Encoding portrait…";
    const input = await ort.Tensor.fromImage($("aligned-cnv").getContext("2d").getImageData(0, 0, 256, 256), {
      tensorFormat: "RGB", tensorLayout: "NCHW"
    });
    for (let i = 0; i < input.data.length; i++) input.data[i] = input.data[i] * 2 - 1;
    const output = await encoder.run({ image: input });
    if (version !== uploadVersion) return;
    const encoded = output[encoder.outputNames[0]];
    if (encoded.data.length !== 18 * 512) throw new Error(`Unexpected latent shape: ${encoded.dims}`);
    baseLatent = new Float32Array(encoded.data);
    $("busy-state").textContent = "Rendering portrait…";
    message(motionSeen ? "Move your phone to edit the portrait." : "Portrait ready. Enable motion to steer it.");
    queueRender();
  } catch (error) {
    console.error("Photo preparation failed:", error);
    if (version === uploadVersion) {
      $("busy-state").hidden = true;
      message(error.message);
    }
  } finally {
    URL.revokeObjectURL(url);
    event.target.value = "";
  }
}

function alignFaceToCanvas(image, keypoints, canvas) {
  const average = indices => ({
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
  const axis = { x: eyeToEye.x + eyeToMouth.y, y: eyeToEye.y - eyeToMouth.x };
  const axisLength = Math.hypot(axis.x, axis.y);
  if (!axisLength) throw new Error("Could not align this face.");
  const radius = Math.max(Math.hypot(eyeToEye.x, eyeToEye.y) * 2, Math.hypot(eyeToMouth.x, eyeToMouth.y) * 1.8);
  const x = { x: axis.x * radius / axisLength, y: axis.y * radius / axisLength };
  const y = { x: -x.y, y: x.x };
  const center = { x: eyeAvg.x + eyeToMouth.x * 0.1, y: eyeAvg.y + eyeToMouth.y * 0.1 };
  const half = canvas.width / 2;
  const outputToSource = new DOMMatrix([x.x / half, x.y / half, y.x / half, y.y / half,
    center.x - x.x - y.x, center.y - x.y - y.y]);
  const sourceToOutput = outputToSource.inverse();
  const context = canvas.getContext("2d");
  context.save();
  context.resetTransform();
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.setTransform(sourceToOutput.a, sourceToOutput.b, sourceToOutput.c,
    sourceToOutput.d, sourceToOutput.e, sourceToOutput.f);
  const corners = [
    { x: center.x - x.x - y.x, y: center.y - x.y - y.y },
    { x: center.x - x.x + y.x, y: center.y - x.y + y.y },
    { x: center.x + x.x + y.x, y: center.y + x.y + y.y },
    { x: center.x + x.x - y.x, y: center.y + x.y - y.y }
  ];
  if (corners.some(point => point.x < 0 || point.x > image.naturalWidth || point.y < 0 || point.y > image.naturalHeight)) {
    for (const [tx, sx] of [[0, -1], [0, 1], [2 * image.naturalWidth, -1]]) {
      for (const [ty, sy] of [[0, -1], [0, 1], [2 * image.naturalHeight, -1]]) {
        context.save();
        context.transform(sx, 0, 0, sy, tx, ty);
        context.drawImage(image, 0, 0);
        context.restore();
      }
    }
  } else context.drawImage(image, 0, 0);
  context.restore();
}

$("img-upload").addEventListener("change", uploadFace);
$("enable-motion").addEventListener("click", enableMotion);
$("recenter").addEventListener("click", resetMotion);
$("strength").addEventListener("input", () => {
  $("strength-value").textContent = $("strength").value;
  queueRender();
});
axisNames.forEach(axis => $(`axis-${axis}`).addEventListener("change", queueRender));
const modelsReady = loadModels();
