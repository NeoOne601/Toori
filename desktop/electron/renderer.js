const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const resultPre = document.getElementById('result');

// Request camera access
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error('Error accessing camera:', err);
    resultPre.textContent = 'Camera access denied.';
  }
}

captureBtn.addEventListener('click', async () => {
  // Draw current video frame to canvas
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert canvas to base64 PNG (you could also send raw Blob)
  const dataUrl = canvas.toDataURL('image/png');
  const base64 = dataUrl.split(',')[1];

  // For this demo we send the raw image data as a placeholder embedding.
  // In a real implementation you would run the MobileViT‑XS encoder (e.g., via a native addon) to get a 128‑dim vector.
  const payload = { embedding: Array(128).fill(0) }; // placeholder

  try {
    const response = await window.electronAPI.search(payload);
    resultPre.textContent = JSON.stringify(response, null, 2);
  } catch (e) {
    console.error('API error:', e);
    resultPre.textContent = 'Error contacting API.';
  }
});

initCamera();
