const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d', { willReadFrequently: true });
const outputMessage = document.getElementById('outputMessage');
const outputData = document.getElementById('outputData');
const cameraSelection = document.getElementById('cameraSelection');

let cvReady = false;
let currentStream = null; // To keep track of the current camera stream
let isPaused = false; // Flag to control pause state

function onOpenCvReady() {
    cv.onRuntimeInitialized = () => {
        console.log("OpenCV ready");
        cvReady = true;
    };
}

// Function to start camera stream
async function startCamera(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            frameRate: { ideal: 60 }
        }
    };

    if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
    } else {
        // If no specific deviceId, prefer environment camera if available
        constraints.video.facingMode = "environment";
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        video.srcObject = stream;
        video.setAttribute("playsinline", true);
        video.play();
        isPaused = false; // Ensure not paused when starting camera
        requestAnimationFrame(tick);
    } catch (err) {
        console.error("Error accessing camera: ", err);
        outputMessage.innerText = "カメラへのアクセスに失敗しました。";
    }
}

// Enumerate devices and populate camera selection dropdown
async function enumerateDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        outputMessage.innerText = "お使いのブラウザはカメラ選択をサポートしていません。";
        return;
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSelection.innerHTML = ''; // Clear existing options
        let defaultDeviceId = null;

        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${cameraSelection.length + 1}`;
                cameraSelection.appendChild(option);

                // Set the first available camera as default
                if (!defaultDeviceId) {
                    defaultDeviceId = device.deviceId;
                }
            }
        });

        if (defaultDeviceId) {
            startCamera(defaultDeviceId);
        }

    } catch (err) {
        console.error("Error enumerating devices: ", err);
        outputMessage.innerText = "カメラデバイスの取得に失敗しました。";
    }
}

// Event listener for camera selection change
cameraSelection.addEventListener('change', (event) => {
    startCamera(event.target.value);
});

// Initial call to enumerate devices and start camera
enumerateDevices();

// Helper function to calculate distance between two points
function dist(p1, p2) {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

// Helper function to order points for perspective transform
function orderPoints(pts) {
    const points = pts.map(p => ({ x: p.x, y: p.y }));
    points.sort((a, b) => (a.x + a.y) - (b.x + b.y));
    const topLeft = points[0];
    const bottomRight = points[3];

    points.sort((a, b) => (a.x - a.y) - (b.x - b.y));
    const topRight = points[0];
    const bottomLeft = points[3];

    return [topLeft, topRight, bottomRight, bottomLeft];
}

function tick() {
    if (isPaused) return; // If paused, do not process frames

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.hidden = false;
        canvas.height = video.videoHeight;
        canvas.width = video.videoWidth;
        context.drawImage(video, 0, 0, canvas.width, canvas.height); // Draw video frame to main canvas

        let code = null;
        let src = null;
        let gray = null;
        let blurred = null;
        let thresh = null;
        let contours = null;
        let hierarchy = null;
        let warped = null;
        let tempCanvas = null;
        let tempContext = null;

        try {
            // --- First attempt: Direct jsQR scan on the original frame (fastest) ---
            let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
            code = jsQR(imageData.data, imageData.width, imageData.height, { inversionAttempts: "dontInvert" });

            if (code) {
                // If QR code found directly, draw its bounding box and display data
                drawLine(code.location.topLeftCorner, code.location.topRightCorner, "#FF3B58");
                drawLine(code.location.topRightCorner, code.location.bottomRightCorner, "#FF3B58");
                drawLine(code.location.bottomRightCorner, code.location.bottomLeftCorner, "#FF3B58");
                drawLine(code.location.bottomLeftCorner, code.location.topLeftCorner, "#FF3B58");
                outputMessage.hidden = true;
                outputData.parentElement.hidden = false;
                outputData.innerText = code.data;

                // Pause video and processing for 1 second
                video.pause();
                isPaused = true;
                setTimeout(() => {
                    video.play();
                    isPaused = false;
                    requestAnimationFrame(tick); // Resume processing after pause
                }, 1000);

            } else if (cvReady) {
                // --- Second attempt: Use OpenCV for advanced QR code detection ---
                src = cv.imread(canvas);
                gray = new cv.Mat();
                blurred = new cv.Mat();
                thresh = new cv.Mat();

                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
                cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
                cv.adaptiveThreshold(blurred, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

                contours = new cv.MatVector();
                hierarchy = new cv.Mat();
                cv.findContours(thresh, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

                let finderPatterns = [];

                // Iterate through contours to find potential finder patterns
                for (let i = 0; i < contours.size(); ++i) {
                    let cnt = contours.get(i);
                    let peri = cv.arcLength(cnt, true);
                    let approx = new cv.Mat();
                    cv.approxPolyDP(cnt, approx, 0.04 * peri, true); // Relaxed epsilon

                    // Check if it's a 4-sided polygon (square-like)
                    if (approx.rows === 4) {
                        let rect = cv.boundingRect(approx);
                        let aspectRatio = rect.width / rect.height;

                        // Check aspect ratio for square-like objects (more relaxed)
                        if (aspectRatio > 0.5 && aspectRatio < 1.5) {
                            // Check for concentric squares (hierarchy check)
                            // A finder pattern has 3 nested contours (outer black, middle white, inner black)
                            // hierarchy[i][2] is the index of the first child contour
                            // hierarchy[i][3] is the index of the parent contour
                            let child1 = hierarchy.data32S[i * 4 + 2];
                            if (child1 !== -1) { // Has a child
                                let child2 = hierarchy.data32S[child1 * 4 + 2];
                                if (child2 !== -1) { // Has a grandchild
                                    // This is a potential finder pattern
                                    finderPatterns.push({
                                        contour: cnt.clone(),
                                        center: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
                                        size: Math.max(rect.width, rect.height)
                                    });
                                }
                            }
                        }
                    }
                    approx.delete();
                }

                // Try to find 3 finder patterns that form a QR code
                let qrCorners = null;
                if (finderPatterns.length >= 3) {
                    // Sort finder patterns by size (largest first) to prioritize prominent ones
                    finderPatterns.sort((a, b) => b.size - a.size);

                    // Iterate through combinations of 3 finder patterns
                    for (let i = 0; i < finderPatterns.length; i++) {
                        for (let j = i + 1; j < finderPatterns.length; j++) {
                            for (let k = j + 1; k < finderPatterns.length; k++) {
                                let p1 = finderPatterns[i];
                                let p2 = finderPatterns[j];
                                let p3 = finderPatterns[k];

                                // Calculate distances between centers
                                let d12 = dist(p1.center, p2.center);
                                let d13 = dist(p1.center, p3.center);
                                let d23 = dist(p2.center, p3.center);

                                // Check for right angle (Pythagorean theorem) and relative distances
                                // The two shorter sides should be roughly equal and form a right angle
                                let sides = [d12, d13, d23].sort((a, b) => a - b);
                                let s1 = sides[0];
                                let s2 = sides[1];
                                let s3 = sides[2];

                                // Check if it forms a right-angled triangle (approximate)
                                if (Math.abs(s1 * s1 + s2 * s2 - s3 * s3) < (0.1 * s3 * s3)) { // Tolerance for right angle
                                    // Check if the two shorter sides are roughly equal
                                    if (Math.abs(s1 - s2) < (0.2 * s1)) { // Tolerance for equal sides
                                        // These three points are likely the finder patterns
                                        // The point opposite to the hypotenuse is the top-left corner
                                        let cornerPoints = [p1.center, p2.center, p3.center];
                                        let tlCandidate = null;
                                        if (d12 === s3) tlCandidate = p3.center;
                                        else if (d13 === s3) tlCandidate = p2.center;
                                        else tlCandidate = p1.center;

                                        // Order the three points to find top-left, top-right, bottom-left
                                        let otherPoints = cornerPoints.filter(p => p !== tlCandidate);
                                        let trCandidate = null;
                                        let blCandidate = null;

                                        if (dist(tlCandidate, otherPoints[0]) < dist(tlCandidate, otherPoints[1])) {
                                            trCandidate = otherPoints[0];
                                            blCandidate = otherPoints[1];
                                        } else {
                                            trCandidate = otherPoints[1];
                                            blCandidate = otherPoints[0];
                                        }

                                        // Calculate the fourth corner (bottom-right) based on the other three
                                        let brCandidate = {
                                            x: trCandidate.x + blCandidate.x - tlCandidate.x,
                                            y: trCandidate.y + blCandidate.y - tlCandidate.y
                                        };

                                        qrCorners = orderPoints([tlCandidate, trCandidate, brCandidate, blCandidate]);
                                        break; // Found QR corners, exit loops
                                    }
                                }
                            }
                            if (qrCorners) break;
                        }
                        if (qrCorners) break;
                    }
                }

                if (qrCorners) {
                    let [tl, tr, br, bl] = qrCorners;

                    // Calculate the width and height of the new image
                    let widthA = dist(br, bl);
                    let widthB = dist(tr, tl);
                    let maxWidth = Math.max(widthA, widthB);

                    let heightA = dist(tr, br);
                    let heightB = dist(tl, bl);
                    let maxHeight = Math.max(heightA, heightB);

                    let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]);
                    let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, maxWidth, 0, maxWidth, maxHeight, 0, maxHeight]);

                    let M = cv.getPerspectiveTransform(srcTri, dstTri);
                    let dsize = new cv.Size(maxWidth, maxHeight);
                    warped = new cv.Mat();
                    cv.warpPerspective(src, warped, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

                    // Create a temporary canvas for jsQR to read from
                    tempCanvas = document.createElement('canvas');
                    tempCanvas.width = warped.cols;
                    tempCanvas.height = warped.rows;
                    tempContext = tempCanvas.getContext('2d');
                    cv.imshow(tempCanvas, warped); // Draw warped image to temp canvas

                    let warpedImageData = tempContext.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                    code = jsQR(warpedImageData.data, warpedImageData.width, warpedImageData.height, { inversionAttempts: "attemptBoth" });

                    // Draw the detected contour on the main canvas
                    // Draw the four corners of the QR code on the original frame
                    drawLine(tl, tr, "#00FF00");
                    drawLine(tr, br, "#00FF00");
                    drawLine(br, bl, "#00FF00");
                    drawLine(bl, tl, "#00FF00");
                    cv.imshow(canvas, src); // Show video frame with contour on main canvas

                    if (code) {
                        outputMessage.hidden = true;
                        outputData.parentElement.hidden = false;
                        outputData.innerText = code.data;

                        // Pause video and processing for 1 second
                        video.pause();
                        isPaused = true;
                        setTimeout(() => {
                            video.play();
                            isPaused = false;
                            requestAnimationFrame(tick); // Resume processing after pause
                        }, 1000);

                    } else {
                        outputMessage.hidden = false;
                        outputData.parentElement.hidden = true;
                        outputMessage.innerText = "QRコードをスキャンしてください...";
                    }

                } else {
                    // If no QR code found by OpenCV, display default message
                    outputMessage.hidden = false;
                    outputData.parentElement.hidden = true;
                    outputMessage.innerText = "QRコードをスキャンしてください...";
                }
            } else {
                // If OpenCV is not ready, display default message
                outputMessage.hidden = false;
                outputData.parentElement.hidden = true;
                outputMessage.innerText = "QRコードをスキャンしてください...";
            }

        } catch (err) {
            console.error("OpenCV or QR scan error: ", err);
            outputMessage.hidden = false;
            outputData.parentElement.hidden = true;
            outputMessage.innerText = "エラーが発生しました。";
        } finally {
            // Memory cleanup
            if (src) src.delete();
            if (gray) gray.delete();
            if (blurred) blurred.delete();
            if (thresh) thresh.delete();
            if (contours) contours.delete();
            if (hierarchy) hierarchy.delete();
            // bestCnt is cloned, so it needs to be deleted if it was created
            // if (bestCnt) bestCnt.delete(); // This is handled inside the loop now
            if (warped) warped.delete();
            // srcTri, dstTri, M are deleted inside the if (qrCorners) block
        }
    }
    if (!isPaused) { // Only request next frame if not paused
        requestAnimationFrame(tick);
    }
}

function drawLine(begin, end, color) {
    context.beginPath();
    context.moveTo(begin.x, begin.y);
    context.lineTo(end.x, end.y);
    context.lineWidth = 4;
    context.strokeStyle = color;
    context.stroke();
}