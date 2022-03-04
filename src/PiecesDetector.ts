const cv = window.cv

export type Piece = {
  x: number,
  y: number,
  width: number,
  height: number,
  image: ImageData,
}

export class PiecesDetector {

  #srcCanvasId: string
  #srcCanvas: HTMLCanvasElement
  #processSizeRatio = 1
  #contourComplexityThreshold = 20
  #minBoxSize = 50

  constructor() {
    this.#srcCanvasId = "srcCanvas-" + Math.floor(Math.random() * 100000)
    this.#srcCanvas = document.createElement("canvas")
    this.#srcCanvas.id = this.#srcCanvasId
    document.body.appendChild(this.#srcCanvas)
  }

  setProcessRatio(ratio: number) {
    this.#processSizeRatio = ratio
  }

  process (videoCanvas: HTMLCanvasElement) {
    this.#srcCanvas.width = videoCanvas.width * this.#processSizeRatio
    this.#srcCanvas.height = videoCanvas.height * this.#processSizeRatio
    const srcContext = this.#srcCanvas.getContext("2d")!
    srcContext.drawImage(
      videoCanvas,
      0, 0, videoCanvas.width, videoCanvas.height,
      0, 0, this.#srcCanvas.width, this.#srcCanvas.height
    )
    let src = cv.imread(this.#srcCanvasId);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(src, src, 100, 255, cv.THRESH_OTSU);
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(src, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    const boxes: Piece[] = []

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i)
      if (contour.size().height >= this.#contourComplexityThreshold) {
        let minX = 10000000
        let maxX = 0
        let minY = 10000000
        let maxY = 0
        for (let j = 0; j < contour.size().height; j++) {
          const x = contour.data32S[j * 2]
          const y = contour.data32S[j * 2 + 1]
          minX = Math.min(minX, x)
          maxX = Math.max(maxX, x)
          minY = Math.min(minY, y)
          maxY = Math.max(maxY, y)
        }

        const x = minX / this.#processSizeRatio
        const y = minY / this.#processSizeRatio
        const width = (maxX - minX) / this.#processSizeRatio
        const height = (maxY - minY) / this.#processSizeRatio

        if (width > this.#minBoxSize && height > this.#minBoxSize) {
          const tmpCanvas = document.createElement("canvas")
          const tmpContext = tmpCanvas.getContext("2d")!
          tmpCanvas.width = width
          tmpCanvas.height = height
          tmpContext.drawImage(
            videoCanvas, x, y, width, height,
            0, 0, tmpCanvas.width, tmpCanvas.height
          )
          tmpContext.globalCompositeOperation = "destination-in"
          tmpContext.beginPath();
          for (let j = 0; j < contour.size().height; j++) {
            const px = contour.data32S[j * 2] / this.#processSizeRatio - x
            const py = contour.data32S[j * 2 + 1] / this.#processSizeRatio - y
            if (j == 0) {
              tmpContext.moveTo(px, py);
            } else {
              tmpContext.lineTo(px, py);
            }
          }
          tmpContext.fillStyle = "red"
          tmpContext.closePath();
          tmpContext.fill();
          boxes.push({
            x,
            y,
            width,
            height,
            image: tmpContext.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height),
          })
        }
      }
    }
    src.delete();
    return boxes
  }
}