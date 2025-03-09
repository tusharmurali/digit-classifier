const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// Load our model.
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./digit_recognition_model.onnx");

// Add 'Draw a number here!' to the canvas.
ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#dedede";
// ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// Set the line color for the canvas.
ctx.strokeStyle = "#ffffff";

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col";
    element.children[0].children[0].style.height = "0";
  }
}

function drawLine(fromX, fromY, toX, toY) {
  // Draws a line from (fromX, fromY) to (toX, toY).
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

async function updatePredictions() {
    const CANVAS_SIZE = 280; // 280x280 canvas size
    const BLOCK_SIZE = 10;   // Each block of 10x10 pixels is averaged to form a single pixel in the 28x28 image
    const GRID_SIZE = 28;    // 28x28 grid size

    // Get the predictions for the canvas data.
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Convert the image data into a 28x28 image (averaging each 10x10 block)
    const processedImage = processImageData(imgData, CANVAS_SIZE, BLOCK_SIZE, GRID_SIZE);

    // Create a tensor for the ONNX model (flattened 28x28 image to 1x784)
    const input = new onnx.Tensor(new Float32Array(processedImage), "float32", [1, 784]);

    // Run the prediction with the model
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    console.log(predictions)

    // Find the max prediction probability
    const maxPrediction = Math.max(...predictions);

    // Update the UI with the prediction results
    predictions.forEach((prediction, i) => {
        const element = document.getElementById(`prediction-${i}`);
        const height = `${prediction * 100}%`;
        element.children[0].children[0].style.height = height;
        element.className = prediction === maxPrediction ? "prediction-col top-prediction" : "prediction-col";
    });
}

// Function to process the image data and average over 10x10 blocks
function processImageData(imgData, canvasSize, blockSize, gridSize) {
    const normalizedData = [];

    // Loop through each 10x10 block
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            let r = 0, g = 0, b = 0, a = 0;
            let count = 0;

            // Average the pixels within each 10x10 block
            for (let dy = 0; dy < blockSize; dy++) {
                for (let dx = 0; dx < blockSize; dx++) {
                    const pixelIndex = ((y * blockSize + dy) * canvasSize + (x * blockSize + dx)) * 4;
                    r += imgData.data[pixelIndex];
                    g += imgData.data[pixelIndex + 1];
                    b += imgData.data[pixelIndex + 2];
                    a += imgData.data[pixelIndex + 3];
                    count++;
                }
            }

            r = r / count;
            g = g / count;
            b = b / count;
            a = a / count;

            // Convert the average to grayscale (using luminosity formula)
            const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Normalize the grayscale value to be between -1 and 1
            const normalizedGray = (gray / 255) * 2 - 1;
            normalizedData.push(normalizedGray);
        }
    }

    // Return the flattened array (28x28 image to 1x784 tensor)
    return normalizedData;
}

function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // To draw a dot on the mouse down event, we set laxtX and lastY to be
  // slightly offset from x and y, and then we call `canvasMouseMove(event)`,
  // which draws a line from (laxtX, lastY) to (x, y) that shows up as a
  // dot because the difference between those points is so small. However,
  // if the points were the same, nothing would be drawn, which is why the
  // 0.001 offset is added.
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  // We won't be able to detect a MouseUp event if the mouse has moved
  // ouside the window, so when the mouse leaves the window, we set
  // `isMouseDown` to false automatically. This prevents lines from
  // continuing to be drawn when the mouse returns to the canvas after
  // having been released outside the window.
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

loadingModelPromise.then(() => {
  canvas.addEventListener("mousedown", canvasMouseDown);
  canvas.addEventListener("mousemove", canvasMouseMove);
  document.body.addEventListener("mouseup", bodyMouseUp);
  document.body.addEventListener("mouseout", bodyMouseOut);
  clearButton.addEventListener("mousedown", clearCanvas);

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})