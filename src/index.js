import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import "./styles.css";
tf.setBackend('webgl');

const threshold = 0.7;

async function load_model() {
  // It's possible to load the model locally or from a repo
  // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json" just set it before in your https server
  const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
  //const model = await loadGraphModel("https://raw.githubusercontent.com/trangml/futureHAUS-TFJS-object-detection/master/models/tf2_web_model/model.json");
  return model;
}

let classesDir = {
  1: {
    name: 'Inverter',
    id: 1,
  },
  2: {
    name: 'Charge Controller',
    id: 2,
  },
  3: {
    name: 'Battery',
    id: 3,
  },
  4: {
    name: 'Other',
    id: 4,
  }
}

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();


  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
    tf.engine().startScope();
    model.executeAsync(this.process_input(video)).then(predictions => {
      this.renderPredictions(predictions, video);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  };

  process_input(video_frame) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  };
  buildDetectedObjectsTF1(num_detections, scores, threshold,
    boxes, classes, classesDir) {
    const detectionObjects = []
    var video_frame = document.getElementById('frame');
    for (var i = 0; i < num_detections[0]; i++) {
      if (scores[0][i] > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[0][i],
          label: classesDir[classes[0][i]].name,
          score: scores[0][i].toFixed(4),
          bbox: bbox
        })
      }
    }
    return detectionObjects
  }
  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = []
    var video_frame = document.getElementById('frame');
    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }

  renderPredictions = predictions => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    // These indexes work for the tf1 model converted using tfjs_wiz 3.9
    // const num_detections = predictions[0].dataSync();
    // const boxes = predictions[5].arraySync();
    // const scores = predictions[1].arraySync();
    // const classes = predictions[4].arraySync();
    // const detections = this.buildDetectedObjectsTF1(num_detections, scores, threshold,
    // boxes, classes, classesDir);

    // These indexes work for the tf2 model converted using tfjs_wiz 3.9
    // const boxes = predictions[7].arraySync(); // name "detection_boxes"
    // const scores = predictions[2].arraySync(); // name "detection_multiclass_scores"
    // const classes = predictions[6].dataSync(); // name "Identity_2:0", shape "tensorShape": { "dim": [{ "size": "1" }, { "size": "100" }] }
    // const detections = this.buildDetectedObjects(scores, threshold,
    // boxes, classes, classesDir);

    // These indexes work for the inverter_long_web model
    // const boxes = predictions[7].arraySync(); // name "detection_boxes"
    // const scores = predictions[6].arraySync(); // name "detection_multiclass_scores"
    // const classes = predictions[3].dataSync(); // name "Identity_2:0", shape "tensorShape": { "dim": [{ "size": "1" }, { "size": "100" }] }
    const p0 = predictions[0].arraySync(); // name "detection_boxes"
    const p1 = predictions[1].arraySync(); // name "detection_boxes"
    const p2 = predictions[2].arraySync(); // name "detection_boxes"
    const p3 = predictions[3].arraySync(); // name "detection_boxes"
    const p4 = predictions[4].arraySync(); // name "detection_boxes"
    const p5 = predictions[5].arraySync(); // name "detection_boxes"
    const p6 = predictions[6].arraySync(); // name "detection_multiclass_scores"
    const p7 = predictions[7].arraySync(); // name "detection_multiclass_scores"
    // These indexes work for the web model trained locally using resnet50
    const boxes = predictions[0].arraySync(); // name "detection_boxes"
    const scores = predictions[7].arraySync(); // name "detection_multiclass_scores"
    const classes = predictions[5].dataSync(); // name "Identity_2:0", shape "tensorShape": { "dim": [{ "size": "1" }, { "size": "100" }] }
    const detections = this.buildDetectedObjects(scores, threshold,
      boxes, classes, classesDir);

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%", x, y);
    });
  };

  render() {
    return (
      <div>
        <h1>Real-Time Object Detection: FutureHAUS</h1>
        <h3>MobileNetV2</h3>
        <video
          style={{ height: '600px', width: "500px" }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="600"
          height="500"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);

