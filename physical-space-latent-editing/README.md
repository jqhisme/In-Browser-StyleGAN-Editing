# Physical space latent editing

Open `index.html` through an HTTPS web server on your phone. Motion sensors are not available to pages served over plain HTTP. On iPhone, tap **Enable motion** to request access, then wait for **Receiving motion** before moving the phone.

The page aligns and encodes an uploaded face, then applies up to three directions from the Hugging Face repository's `latent_directions.json` to all 18 StyleGAN latent layers. Model weights are loaded from the same repository, and inference runs in the browser.

The X, Y, and Z controls use short accelerometer gestures in the phone's portrait coordinate frame. Acceleration alone cannot measure absolute position accurately, so the values are bounded and may drift. Tap **Recenter** to reset them. The page falls back to WASM when WebGPU is unavailable; initial model loading and each render can be slow on a phone.
