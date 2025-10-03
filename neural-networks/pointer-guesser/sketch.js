var nn;
var mode = "collect"; // "collect" eller "predict"
var isTrained = false;
var predicting = false;
var pending = false; // throttle classify
var currentLabel = "UL";
var counts = { UL:0, UR:0, LL:0, LR:0 };
var lastPrediction = { label: "—", confidence: 0 };

var W = 420, H = 420;

async function setup() {

  ml5.setBackend("webgl")
  var cnv = createCanvas(W, H);
  cnv.parent("sketch");
  textFont("system-ui");
  resetNN();

  document.getElementById("labelSelect").addEventListener("change", function (e) {
    currentLabel = e.target.value;
  });

  document.getElementById("trainBtn").addEventListener("click", trainModel);

  document.getElementById("predictBtn").addEventListener("click", function () {
    if (mode === "collect") {
      mode = "predict";
      predicting = true;
      setStatus("predicting…");
    } else {
      mode = "collect";
      predicting = false;
      setStatus("idle");
    }
  });
}

function resetNN() {
  nn = ml5.neuralNetwork({
    inputs: ["x", "y"],
    outputs: ["label"],
    task: "classification",
    debug: true // tfjs-vis grafer vid träning
  });
  counts = { UL:0, UR:0, LL:0, LR:0 };
  updateCounts();
  lastPrediction = { label: "—", confidence: 0 };
  isTrained = false;
  predicting = false;
  mode = "collect";
  pending = false;
  setStatus("idle");
}

function mousePressed() {
  // Samla BARA data i collect-läge
  if (mode !== "collect") return;
  if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) return;
  var x = mouseX / width;
  var y = mouseY / height;
  nn.addData({ x: x, y: y }, { label: currentLabel });
  counts[currentLabel]++;
  updateCounts();
}

function trainModel() {
  isTrained = false;
  setStatus("normalizing…");
  nn.normalizeData();
  setStatus("training…");

  nn.train(
    { epochs: 30, batchSize: 16 },
    function onEpoch(epoch, loss) {
      var lossTxt = (loss && loss.loss && loss.loss.toFixed) ? loss.loss.toFixed(4) : "-";
      setStatus("training… epoch " + epoch + " loss: " + lossTxt);
    },
    function onDone() {
      isTrained = true;
      // Gå automatisk till predict-läge direkt
      mode = "predict";
      predicting = true;
      setStatus("trained ✔  predicting…");

      // Sanity check: testa en punkt i mitten och logga resultat
      testOnePoint(0.5, 0.5);
    }
  );
}

async function testOnePoint(nx, ny) {
  try {
    var results = await nn.classify({ x: nx, y: ny });
    console.log('[TEST classify(0.5,0.5)] results = ', results);
  } catch (e) {
    console.error('[TEST classify] error:', e);
  }
}

async function classifyPoint(px, py) {
  if (!isTrained || pending) return;
  pending = true;
  try {
    var results = await nn.classify({ x: px/width, y: py/height });
    if (results && results[0]) {
      var r = results[0];
      var confVal = (typeof r.confidence === "number") ? r.confidence : 0;
      lastPrediction.label = r.label;
      lastPrediction.confidence = confVal;
      // Debug-logga lätt för att se att det tickar
      // console.log('pred:', r.label, confVal);
    } else {
      console.warn('No results from classify');
    }
  } catch (err) {
    console.error('classify error:', err);
  } finally {
    pending = false;
  }
}

function draw() {
  background(255);
  // Quadrant-linjer
  stroke(220); strokeWeight(2);
  line(width/2, 0, width/2, height);
  line(0, height/2, width, height/2);

  // Hörnlabels
  noStroke(); fill(180); textSize(14);
  textAlign(LEFT, TOP); text("UL", 8, 8);
  textAlign(RIGHT, TOP); text("UR", width-8, 8);
  textAlign(LEFT, BOTTOM); text("LL", 8, height-8);
  textAlign(RIGHT, BOTTOM); text("LR", width-8, height-8);

  // Musmarkör
  noFill(); stroke(80); circle(mouseX, mouseY, 14);

  // Endast i predict-läge + tränad modell
  if (mode === "predict" && isTrained) {
    // throttla till var 5:e frame
    if (frameCount % 5 === 0) classifyPoint(mouseX, mouseY);

    noStroke(); fill(30); textAlign(LEFT, BOTTOM); textSize(16);
    var conf = Math.round((lastPrediction.confidence || 0) * 100);
    var confTxt = isNaN(conf) ? "—" : (conf + "%");
    text("Prediktion: " + lastPrediction.label + " (" + confTxt + ")", 10, height - 36);
  }
}

function updateCounts() {
  document.getElementById("ulCount").textContent = counts.UL;
  document.getElementById("urCount").textContent = counts.UR;
  document.getElementById("llCount").textContent = counts.LL;
  document.getElementById("lrCount").textContent = counts.LR;
}

function setStatus(msg) {
  document.getElementById("status").textContent = "status: " + msg;
}
