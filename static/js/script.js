/* Canvas & Drawing Logic */
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Setup Canvas
ctx.fillStyle = "white"; // MNIST uses white text on black, or we invert. 
// Let's make background white and draw black, then invert in backend.
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoords(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const [x, y] = getCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    [lastX, lastY] = [x, y];
}

function getCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return [clientX - rect.left, clientY - rect.top];
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch Support
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

/* Tab Switching */
function switchMode(mode) {
    const drawArea = document.getElementById('draw-area');
    const uploadArea = document.getElementById('upload-area');
    const btnDraw = document.getElementById('btn-draw-mode');
    const btnUpload = document.getElementById('btn-upload-mode');

    if (mode === 'draw') {
        drawArea.style.display = 'block';
        uploadArea.style.display = 'none';
        btnDraw.classList.add('active-tab');
        btnDraw.classList.remove('btn-secondary');
        btnDraw.classList.add('btn-primary');

        btnUpload.classList.remove('active-tab', 'btn-primary');
        btnUpload.classList.add('btn-secondary');
    } else {
        drawArea.style.display = 'none';
        uploadArea.style.display = 'block';
        btnUpload.classList.add('active-tab');
        btnUpload.classList.remove('btn-secondary');
        btnUpload.classList.add('btn-primary');

        btnDraw.classList.remove('active-tab', 'btn-primary');
        btnDraw.classList.add('btn-secondary');
    }
}

// Set initial state
switchMode('draw');

/* API Prediction */
async function predictDrawing() {
    const image = canvas.toDataURL('image/png');
    sendPredictionRequest(image, 'predict-btn');
}

async function predictUpload() {
    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        sendPredictionRequest(e.target.result, 'upload-predict-btn');
    };
    reader.readAsDataURL(file);
}

async function sendPredictionRequest(base64Image, btnId) {
    const btn = document.getElementById(btnId);
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    // Loading State
    btn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data.prediction, data.confidence);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to server.');
    } finally {
        // Reset State
        btn.disabled = false;
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
}

function displayResult(prediction, confidence) {
    const resultBox = document.getElementById('result-box');
    const resultContent = document.getElementById('result-content');

    // Hide placeholder
    // Show content
    resultBox.innerHTML = '';
    resultBox.appendChild(resultContent);
    resultContent.style.display = 'block';

    document.getElementById('prediction-digit').innerText = prediction;
    document.getElementById('confidence-text').innerText = confidence;
    document.getElementById('confidence-bar-fill').style.width = confidence;
}

/* Drag and Drop Upload */
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const uploadBtn = document.getElementById('upload-predict-btn');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', handleFiles);

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFiles();
    }
});

function handleFiles() {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}
