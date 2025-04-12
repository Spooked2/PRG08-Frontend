//Imports
import {FilesetResolver, PoseLandmarker, DrawingUtils} from '@mediapipe/tasks-vision';

//Wait for the document to load before running any functions that require DOM elements
window.addEventListener('load', init);
ml5.setBackend("webgl");

//Variables
const nnOptions = {task: 'classification', debug: true,}
const performanceNN = ml5.neuralNetwork(nnOptions);
const poseNN = ml5.neuralNetwork(nnOptions);
const studentTemplate = {
    Age: 15,
    Gender: 0,
    Ethnicity: 0,
    ParentalEducation: 0,
    StudyTimeWeekly: 0,
    Absences: 0,
    Tutoring: 0,
    ParentalSupport: 0,
    Extracurricular: 0,
    Sports: 0,
    Music: 0,
    Volunteering: 0,
    GPA: 0
}

let videoElement;
let canvas;
let webcamControlContainer;
let webcamContainer;

let poseLandmarker;
let isWebcamRunning = false;
let canvasContext;
let drawingUtils;

let startTimer;
let form;
let feedbackContainer;

//Functions
async function init() {

    //Get all the DOM elements we need
    videoElement = document.getElementById('webcamElement');
    canvas = document.getElementById('landmarkOverlay');
    webcamControlContainer = document.getElementById('webcamControls');
    webcamContainer = document.getElementById('webcamContainer');
    form = document.getElementById('studentForm');
    feedbackContainer = document.getElementById('feedbackContainer');

    //Add an event listener to the form so we can do things upon it's submission
    form.addEventListener("submit", submitHandler);

    //Get the canvas context
    canvasContext = canvas.getContext('2d');

    drawingUtils = new DrawingUtils(canvasContext);

    //Clear out the canvas just in case
    clearCanvas();

    //Load the NN models
    try {

        const studentPerformance = {
            model: './models/studentPerformanceModel.json',
            metadata: './models/studentPerformanceModel_meta.json',
            weights: './models/studentPerformanceModel.weights.bin'
        }

        await performanceNN.load(studentPerformance, () => console.log("Student performance successfully loaded!"));

        const poseDetection = {
            model: './models/poseDetection.json',
            metadata: './models/poseDetection_meta.json',
            weights: './models/poseDetection.weights.bin'
        }

        await poseNN.load(poseDetection, () => console.log("Pose detection successfully loaded!"));

        //Wait for the pose landmarker to be created before allowing the buttons to work
        await createPoseLandmarker();

        //Add event listeners to buttons
        webcamControlContainer.addEventListener('click', controlWebcam);

        //TODO: Un-hide the page once it's done loading
        document.getElementById('homePage').classList.toggle('hidden');

    } catch (error) {
        console.error(error.message);

        //TODO: Show the error to the user

    }

}

async function createPoseLandmarker() {

    const vision = await FilesetResolver.forVisionTasks('./node_modules/@mediapipe/tasks-vision/wasm');

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "./pose_landmarker_lite.task",
            delegate: 'GPU'
        },
        runningMode: 'IMAGE',
        numPoses: 1
    });

}

async function controlWebcam(e) {

    //Don't do anything if the pressed thing wasn't a button
    if (e.target.tagName !== 'BUTTON') {
        return;
    }

    if (e.target.id === 'startWebcam') {

        if (isWebcamRunning) {
            videoElement.play();
            clearCanvas();
            return;
        }

        await startWebcam();

        return;
    }

    if (e.target.id === 'stopWebcam') {

        videoElement.srcObject.getTracks()[0].stop();

        isWebcamRunning = false;

        return;
    }

    if (e.target.id === 'testDetection') {

        clearTimeout(startTimer);

        startTimer = setTimeout(startDetection, 2000);

    }

}

async function startWebcam() {

    try {

        videoElement.srcObject = await navigator.mediaDevices.getUserMedia({video: true, audio: false});

        videoElement.addEventListener("loadeddata", () => {

            canvas.style.width = videoElement.videoWidth;
            canvas.style.height = videoElement.videoHeight;

            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;

            webcamContainer.style.height = videoElement.videoHeight + "px";

            isWebcamRunning = true;
        });

    } catch (error) {
        console.error(error.message);
    }

}

async function startDetection() {

    videoElement.pause();

    canvasContext.drawImage(videoElement, 0, 0);

    //Reset the timers just in case
    clearTimeout(startTimer);

    startTimer = setTimeout(async () => {

        const pose = detectPose();
        drawPose(pose);
        console.log(pose);

        isWebcamRunning = false;
        videoElement.srcObject.getTracks()[0].stop();

        const simplifiedPose = [];

        for (const landmark of pose) {
            simplifiedPose.push(landmark.x);
            simplifiedPose.push(landmark.y);
            simplifiedPose.push(landmark.z);
        }

        const result = await poseNN.classify(simplifiedPose);

        //TODO: Start doing things based on what was detected
        console.log(result);

    }, 3000);

}

function detectPose() {

    const result = poseLandmarker.detect(videoElement);

    return result.landmarks[0];

}

function clearCanvas() {
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
}

function drawPose(pose) {
    drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {color: "#FFDDEE", lineWidth: 5});
    drawingUtils.drawLandmarks(pose, {radius: 4, color: "#FF00FF", lineWidth: 2});
}

async function submitHandler(e) {

    e.preventDefault();

    //Grab the data from the form
    const formData = new FormData(form);

    //Turn it into an array so we can work with it better
    //This also has the benefit of allowing us to change all the strings into numbers at the same time
    const formDataArray = Array.from(formData);

    let formDataObject = {};

    for (const input of formDataArray) {

        formDataObject[input[0]] = Number(input[1]);

    }

    //Merge the
    const newStudent = {...studentTemplate, ...formDataObject};

    const result = await performanceNN.classify(newStudent);

    let mostConfidentLabel = {confidence: 0};

    for (const label of result) {

        if (label.confidence > mostConfidentLabel.confidence) {
            mostConfidentLabel = label
        }

    }

    let grade = "no grade could be loaded!"

    switch (mostConfidentLabel.label) {
        case('0'):
            grade = 'A';
            break;
        case('1'):
            grade = 'B';
            break;
        case('2'):
            grade = 'C';
            break;
        case('3'):
            grade = 'D';
            break;
        case('4'):
            grade = 'F';
            break;
    }

    //TODO: Show better feedback to the user
    feedbackContainer.innerText = `Your grades probably average around ${grade}`;

}