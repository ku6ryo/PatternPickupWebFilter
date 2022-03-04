import Stats from "stats.js";
import { Piece, PiecesDetector } from "./PiecesDetector";
import { SupportedModels, createDetector, Keypoint } from "@tensorflow-models/hand-pose-detection"
const { MediaPipeHands } = SupportedModels
import { Vector3 } from "./Vector3";

declare global {
  interface Window {
    cv: any;
  }
}

const stats = new Stats()
document.body.appendChild(stats.dom)

function createVector(v0: Keypoint, v1: Keypoint) {
  return new Vector3(v1.x - v0.x, v1.y - v0.y, (v1.z || 0) - (v0.z || 0));
}

async function main() {
  const detector = await createDetector(MediaPipeHands, {
    runtime: "mediapipe",
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1635986972/",
  })
  const mainCanvas = document.createElement("canvas")
  mainCanvas.style.height = "100vh"
  mainCanvas.style.width = "100vw"
  mainCanvas.style.transform = "scale(-1, 1)"
  const mainContext = mainCanvas.getContext("2d")!
  document.querySelector(".container")!.appendChild(mainCanvas)

  const cameraVideo = document.createElement("video");
  const cameraCanvas = document.createElement("canvas");
  cameraCanvas.style.display = "none"
  document.body.appendChild(cameraCanvas)
  cameraVideo.addEventListener("playing", () => {
    const vw = cameraVideo.videoWidth
    const vh = cameraVideo.videoHeight
    mainCanvas.width = vw
    mainCanvas.height = vh
    mainCanvas.style.maxHeight = `calc(100vw * ${vh / vw})`
    mainCanvas.style.maxWidth = `calc(100vh * ${vw / vh})`
    cameraCanvas.width = vw
    cameraCanvas.height = vh
    requestAnimationFrame(process)
  })
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: {
          ideal: 1280
        },
        height: {
          ideal: 720
        }
      },
    })
    .then(function (stream) {
      cameraVideo.srcObject = stream;
      cameraVideo.play();
    })
    .catch(function (e) {
      console.log(e)
      console.log("Something went wrong!");
    });
  } else {
    alert("getUserMedia not supported on your browser!");
  }

  const piecesDetector = new PiecesDetector()
  piecesDetector.setProcessRatio(0.2)

  let frames = 0
  let pieces: Piece[] | null = null
  let holding = false
  let holdingPiece: Piece | null = null

  async function process () {
    stats.begin()
    cameraCanvas.getContext("2d")!.drawImage(cameraVideo, 0, 0)

    if (holdingPiece) {
      mainContext.filter = "grayscale(100%)"
    } else {
      mainContext.filter = "none"
    }
    mainContext.drawImage(cameraVideo, 0, 0)
    mainContext.filter = "none"
    if (frames % 3 === 0) {
      pieces = piecesDetector.process(cameraCanvas)
      /*
      for (let i = 0; i < pieces.length; i++) {
        const box = pieces[i]
        const { x, y, width, height, image} = box
        mainContext.drawImage(
          await createImageBitmap(image),
          0, 0, image.width, image.height,
          x, y, width, height
        )
        mainContext.strokeStyle = "red"
        mainContext.strokeRect(x, y, width, height)
      }
      */
    }

    const hands = await detector.estimateHands(cameraCanvas)
    if (hands.length > 0) {
      const hand = hands[0]
      const { keypoints3D: kp3, keypoints: kp } = hand
      if (kp3) {
        const v0 = createVector(kp3[5], kp3[6])
        const tipLength = v0.length()
        const distance = createVector(kp3[8], kp3[4]).length()
        if (distance < tipLength * 2 && pieces && pieces.length > 0) {
          const { x: indexX, y: indexY } = kp[8]
          holding = true
          if (!holdingPiece) {
            let minD = 100000
            pieces.forEach((piece) => {
              const d = Math.sqrt(Math.pow(piece.x - indexX, 2) + Math.pow(piece.y - indexY, 2))
              if (minD > d) {
                minD = d
              }
              if (d < 100) {
                holdingPiece = piece
              }
            })
          }
          if (holdingPiece) {
            const { width, height, image } = holdingPiece
            const bitmap = await createImageBitmap(image)
            mainContext.drawImage(
              bitmap, 0, 0, width, height,
              kp[8].x, kp[8].y, width, height)
          }
        } else {
          holding = false
          holdingPiece = null
        }
      } else {
        holding = false
        holdingPiece = null
      }
    }

    frames += 1

    stats.end()
    requestAnimationFrame(process)
  }
}

main()