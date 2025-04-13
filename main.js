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

const model1 = new URL('./models/studentPerformanceModel.json', import.meta.url).href;
const model1Meta = new URL('./models/studentPerformanceModel_meta.json', import.meta.url).href;
const model1Weights = new URL('./models/studentPerformanceModel.weights.bin', import.meta.url).href;
const model2 = new URL('./models/poseDetection.json', import.meta.url).href;
const model2Meta = new URL('./models/poseDetection_meta.json', import.meta.url).href;
const model2Weights = new URL('./models/poseDetection.weights.bin', import.meta.url).href;

let homePage;
let studentPerformancePage;
let lastPose;
let root;

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
let predictedGradeContainer;

//Functions
async function init() {

    //Get all the DOM elements we need
    videoElement = document.getElementById('webcamElement');
    canvas = document.getElementById('landmarkOverlay');
    webcamControlContainer = document.getElementById('webcamControls');
    webcamContainer = document.getElementById('webcamContainer');
    form = document.getElementById('studentForm');
    feedbackContainer = document.getElementById('feedbackContainer');
    homePage = document.getElementById('homePage');
    studentPerformancePage = document.getElementById('studentPerformancePage');
    predictedGradeContainer = document.getElementById('predictedGradeContainer');
    root = document.querySelector(':root');

    document.getElementById('backButton').addEventListener('click', switchPage);

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
            model: model1,
            metadata: model1Meta,
            weights: model1Weights
        }

        await performanceNN.load(studentPerformance, () => console.log("Student performance successfully loaded!"));

        const poseDetection = {
            model: model2,
            metadata: model2Meta,
            weights: model2Weights
        }

        await poseNN.load(poseDetection, () => console.log("Pose detection successfully loaded!"));

        //Wait for the pose landmarker to be created before allowing the buttons to work
        await createPoseLandmarker();

        //Add event listeners to buttons
        webcamControlContainer.addEventListener('click', controlWebcam);

        homePage.classList.toggle('hidden');

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

        startDetection();

    }

}

async function startWebcam() {

    try {

        videoElement.srcObject = await navigator.mediaDevices.getUserMedia({video: true, audio: false});

        videoElement.addEventListener("loadeddata", () => {

            lastPose = 'none';
            updateStyle();

            webcamContainer.classList.remove('hidden');

            isWebcamRunning = true;
        });

    } catch (error) {
        console.error(error.message);
    }

}

function startDetection() {

    //Prevent the detection of a pose if the webcam isn't running
    if (!isWebcamRunning) {
        return;
    }

    //Reset the timers just in case
    clearTimeout(startTimer);

    startTimer = setTimeout(async () => {

        videoElement.pause();

        const pose = detectPose();

        isWebcamRunning = false;
        videoElement.srcObject.getTracks()[0].stop();

        const simplifiedPose = [];

        for (const landmark of pose) {
            simplifiedPose.push(landmark.x);
            simplifiedPose.push(landmark.y);
            simplifiedPose.push(landmark.z);
        }

        const result = await poseNN.classify(simplifiedPose);

        const mostConfidentLabel = getMostConfidentLabel(result);

        switch (mostConfidentLabel) {

            case ('handsUp'):
                handsUpAction();
                break;

            case ('eyesCovered'):
                eyesCoveredAction();
                break;

            case ('fakeSurprise'):
                fakeSurpriseAction();
                break;

            default:
                console.error(mostConfidentLabel);

        }

        updateStyle();

    }, 3000);

}

function detectPose() {

    const result = poseLandmarker.detect(videoElement);

    return result.landmarks[0];

}

function clearCanvas() {
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
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

    const mostConfidentLabel = getMostConfidentLabel(result);

    let grade = "no grade could be loaded!"

    switch (mostConfidentLabel) {
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

    predictedGradeContainer.innerText = `${grade}`;

}

function getMostConfidentLabel(result) {

    let mostConfidentLabel = {confidence: 0};

    for (const label of result) {

        if (label.confidence > mostConfidentLabel.confidence) {
            mostConfidentLabel = label
        }

    }

    return mostConfidentLabel.label;

}

function switchPage() {

    homePage.classList.toggle('hidden');

    studentPerformancePage.classList.toggle('hidden');

}

function updateStyle() {

    switch (lastPose) {

        case ('handsUp'):
            root.style.setProperty('--background', '#7F7');
            root.style.setProperty('--textColor', '#111');
            feedbackContainer.classList.remove('hidden');
            break;

        case ('eyesCovered'):
            root.style.setProperty('--background', '#111');
            root.style.setProperty('--textColor', '#EEE');
            feedbackContainer.classList.add('hidden');
            break;

        case ('fakeSurprise'):
            root.style.setProperty('--background', '#DD9');
            root.style.setProperty('--textColor', '#111');
            feedbackContainer.classList.add('hidden');
            break;

        default:
            root.style.setProperty('--background', '#DDD');
            root.style.setProperty('--textColor', '#111');
            feedbackContainer.classList.add('hidden');
    }

    webcamContainer.classList.add('hidden');

}

function handsUpAction() {

    lastPose = 'handsUp';

}

function eyesCoveredAction() {

    lastPose = 'eyesCovered';

    window.open('https://aibusiness.com/ml/neural-networks', '_blank').focus();

}

function fakeSurpriseAction() {

    lastPose = 'fakeSurprise';

    switchPage();

}