// Model download progress for the camera-on-face demo. MB uses decimal megabytes (1 MB = 1,000,000 bytes).
export async function loadModelWithProgress(base, filename, id) {
    const progress = document.getElementById(`${id}-progress`);
    const size = document.getElementById(`${id}-size`);
    const status = document.getElementById(`${id}-status`);
    const controller = new AbortController();
    let downloaded = 0;
    let total = null;
    const formatMB = bytes => (bytes / 1_000_000).toFixed(2);
    const update = () => {
        size.textContent = `${formatMB(downloaded)} MB / ${total === null ? "unknown total" : `${formatMB(total)} MB`}`;
        if (total === null) {
            progress.removeAttribute("value");
        } else {
            progress.max = total || 1;
            progress.value = downloaded;
        }
    };

    try {
        status.textContent = "Downloading...";
        update();
        const responses = await Promise.all([filename, `${filename}.data`].map(async path => {
            const response = await fetch(base + path, { signal: controller.signal });
            if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
            return response;
        }));
        // Compressed responses report transfer size, whereas stream chunks are decoded.
        const lengths = responses.map(response => {
            const encoding = response.headers.get("content-encoding");
            const length = Number(response.headers.get("content-length"));
            return (!encoding || encoding === "identity") && length > 0 ? length : null;
        });
        if (lengths.every(length => length !== null)) total = lengths.reduce((sum, length) => sum + length, 0);
        update();

        const buffers = await Promise.all(responses.map(async response => {
            if (!response.body) {
                const buffer = new Uint8Array(await response.arrayBuffer());
                downloaded += buffer.byteLength;
                update();
                return buffer;
            }
            const reader = response.body.getReader();
            const chunks = [];
            let length = 0;
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    length += value.byteLength;
                    downloaded += value.byteLength;
                    update();
                }
            } finally {
                reader.releaseLock();
            }
            const buffer = new Uint8Array(length);
            let offset = 0;
            for (const chunk of chunks) {
                buffer.set(chunk, offset);
                offset += chunk.byteLength;
            }
            return buffer;
        }));
        total = downloaded;
        update();
        status.textContent = "Initializing...";
        const session = await ort.InferenceSession.create(buffers[0], {
            executionProviders: ["webgpu", "wasm"],
            externalData: [{ path: `${filename}.data`, data: buffers[1] }]
        });
        status.textContent = "Loaded";
        return session;
    } catch (error) {
        controller.abort();
        status.textContent = "Failed to load";
        throw error;
    }
}
