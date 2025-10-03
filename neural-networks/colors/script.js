let nn;
let isTrained = false;

function buildNN() {
  nn = ml5.neuralNetwork({
    inputs: ['r','g','b'],
    outputs: ['label'],
    task: 'classification',
    debug: true   // visar tfjs-vis grafer
  });

  ml5.setBackend("webgl");
  
  // Enkel dataset: rena RGB-färger
  nn.addData({r:1, g:0, b:0}, {label:'red'});
  nn.addData({r:0, g:1, b:0}, {label:'green'});
  nn.addData({r:0, g:0, b:1}, {label:'blue'});
}

function setStatus(msg) {
  document.getElementById('status').textContent = "status: " + msg;
}

function trainModel() {
  setStatus("normalizing…");
  nn.normalizeData();
  setStatus("training…");
  nn.train({epochs:40}, () => {
    setStatus("trained ✔");
    isTrained = true;
  });
}

async function predictColor() {
  if (!isTrained) {
    setStatus("Train model first!");
    return;
  }
  const hex = document.getElementById('colorInput').value; // #rrggbb
  const r = parseInt(hex.substr(1,2), 16)/255;
  const g = parseInt(hex.substr(3,2), 16)/255;
  const b = parseInt(hex.substr(5,2), 16)/255;

  const results = await nn.classify({r,g,b});
  if (results && results[0]) {
    const {label, confidence} = results[0];
    document.getElementById('output').textContent =
      `Prediction: ${label} (${Math.round(confidence*100)}%)`;
    setStatus("prediction done");
  }
}

// ----- Init -----
buildNN();
document.getElementById('trainBtn').addEventListener('click', trainModel);
document.getElementById('predictBtn').addEventListener('click', predictColor);
