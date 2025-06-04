class CatastrophicForgettingViz {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.classifier = new LinearClassifier(0.05);
        
        this.xMin = 0;
        this.xMax = 8;
        this.yMin = 0;
        this.yMax = 8;
        
        this.dataset1 = generateDataset1();
        this.dataset2 = generateDataset2();
        this.currentDataset = this.dataset1;
        
        this.phase = 1; // 1: initial, 2: training on dataset2, 3: testing original
        this.isTraining = false;
        this.trainingInterval = null;
        this.phaseTrainingEpochs = 0;
        this.targetEpochs = 50;
        
        this.initializeControls();
        this.draw();
    }

    initializeControls() {
        document.getElementById('train-phase1').addEventListener('click', () => this.trainPhase1());
        document.getElementById('train-phase2').addEventListener('click', () => this.trainPhase2());
        document.getElementById('test-original').addEventListener('click', () => this.testOriginal());
        document.getElementById('reset').addEventListener('click', () => this.reset());
        document.getElementById('step-train').addEventListener('click', () => this.stepTrain());
        document.getElementById('auto-train').addEventListener('click', () => this.autoTrain());
        document.getElementById('pause-train').addEventListener('click', () => this.pauseTrain());
    }

    worldToCanvas(worldX, worldY) {
        const canvasX = ((worldX - this.xMin) / (this.xMax - this.xMin)) * this.canvas.width;
        const canvasY = this.canvas.height - ((worldY - this.yMin) / (this.yMax - this.yMin)) * this.canvas.height;
        return [canvasX, canvasY];
    }

    drawBackground() {
        const testPoints = generateTestPoints(this.xMin, this.xMax, this.yMin, this.yMax, 50);
        
        for (const point of testPoints) {
            const prediction = this.classifier.predict(point);
            const [canvasX, canvasY] = this.worldToCanvas(point[0], point[1]);
            
            const intensity = Math.abs(prediction - 0.5) * 2;
            const alpha = intensity * 0.3;
            
            if (prediction > 0.5) {
                this.ctx.fillStyle = `rgba(0, 100, 255, ${alpha})`;
            } else {
                this.ctx.fillStyle = `rgba(255, 100, 0, ${alpha})`;
            }
            
            this.ctx.fillRect(canvasX - 2, canvasY - 2, 4, 4);
        }
    }

    drawDecisionBoundary() {
        const boundary = this.classifier.getDecisionBoundary(this.xMin, this.xMax, this.yMin, this.yMax);
        
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        if (boundary.type === 'line') {
            const [x1, y1] = this.worldToCanvas(boundary.points[0][0], boundary.points[0][1]);
            const [x2, y2] = this.worldToCanvas(boundary.points[1][0], boundary.points[1][1]);
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
        } else if (boundary.type === 'vertical') {
            const [x, y1] = this.worldToCanvas(boundary.x, this.yMin);
            const [, y2] = this.worldToCanvas(boundary.x, this.yMax);
            this.ctx.moveTo(x, y1);
            this.ctx.lineTo(x, y2);
        }
        
        this.ctx.stroke();
    }

    drawDataPoints(dataset, highlight = false) {
        for (let i = 0; i < dataset.length; i++) {
            const [point, label] = dataset[i];
            const [canvasX, canvasY] = this.worldToCanvas(point[0], point[1]);
            
            this.ctx.beginPath();
            this.ctx.arc(canvasX, canvasY, highlight ? 12 : 8, 0, 2 * Math.PI);
            
            if (label === 0) {
                this.ctx.fillStyle = highlight ? '#ff4444' : '#ff6666';
                this.ctx.strokeStyle = '#cc0000';
            } else {
                this.ctx.fillStyle = highlight ? '#4444ff' : '#6666ff';
                this.ctx.strokeStyle = '#0000cc';
            }
            
            this.ctx.fill();
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            if (highlight) {
                const prediction = this.classifier.predictClass(point);
                const isCorrect = prediction === label;
                
                this.ctx.fillStyle = isCorrect ? '#00aa00' : '#aa0000';
                this.ctx.font = 'bold 12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(isCorrect ? '✓' : '✗', canvasX, canvasY);
            }
        }
    }

    drawAxes() {
        this.ctx.strokeStyle = '#ddd';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i <= 8; i++) {
            const [x, y1] = this.worldToCanvas(i, this.yMin);
            const [, y2] = this.worldToCanvas(i, this.yMax);
            this.ctx.beginPath();
            this.ctx.moveTo(x, y1);
            this.ctx.lineTo(x, y2);
            this.ctx.stroke();
            
            const [x1, y] = this.worldToCanvas(this.xMin, i);
            const [x2, ] = this.worldToCanvas(this.xMax, i);
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y);
            this.ctx.lineTo(x2, y);
            this.ctx.stroke();
        }
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.drawAxes();
        this.drawBackground();
        this.drawDecisionBoundary();
        
        if (this.phase === 1 || this.phase === 2) {
            this.drawDataPoints(this.currentDataset);
        }
        
        if (this.phase === 3) {
            this.drawDataPoints(this.dataset1, true);
        }
        
        this.updateMetrics();
    }

    updateMetrics() {
        const metrics = this.classifier.getMetrics();
        let accuracy = metrics.accuracy;
        
        // Calculate accuracy for testing phase
        if (this.phase === 3) {
            let correct = 0;
            for (const [point, label] of this.dataset1) {
                if (this.classifier.predictClass(point) === label) {
                    correct++;
                }
            }
            accuracy = correct / this.dataset1.length;
        }
        
        const accuracyPercent = accuracy * 100;
        let accuracyClass = 'accuracy-low';
        if (accuracyPercent >= 80) {
            accuracyClass = 'accuracy-high';
        } else if (accuracyPercent >= 50) {
            accuracyClass = 'accuracy-medium';
        }
        
        document.getElementById('epoch-value').textContent = metrics.epoch;
        document.getElementById('accuracy-value').textContent = `${accuracyPercent.toFixed(1)}%`;
        document.getElementById('accuracy-value').className = `metric-value accuracy-value ${accuracyClass}`;
        document.getElementById('loss-value').textContent = metrics.loss.toFixed(3);
        document.getElementById('weights-value').textContent = `[${metrics.weights[0].toFixed(2)}, ${metrics.weights[1].toFixed(2)}]`;
        document.getElementById('bias-value').textContent = metrics.bias.toFixed(2);
    }

    updatePhase(phase, message) {
        this.phase = phase;
        const indicator = document.getElementById('phase-indicator');
        const status = document.getElementById('status');
        
        status.textContent = message;
        
        if (phase === 1) {
            indicator.textContent = 'Phase 1: Training on Dataset 1';
            indicator.className = 'phase-indicator phase-1';
        } else if (phase === 2) {
            indicator.textContent = 'Phase 2: Training on Dataset 2';
            indicator.className = 'phase-indicator phase-2';
        } else if (phase === 3) {
            indicator.textContent = 'Phase 3: Testing Original Data';
            indicator.className = 'phase-indicator phase-3';
        }
    }

    trainPhase1() {
        if (this.isTraining) return;
        
        this.pauseTrain();
        this.currentDataset = this.dataset1;
        this.phaseTrainingEpochs = 0;
        this.updatePhase(1, 'Training on original dataset...');
        
        document.getElementById('train-phase1').disabled = true;
        
        this.isTraining = true;
        this.trainingInterval = setInterval(() => {
            this.classifier.trainStep(this.currentDataset);
            this.phaseTrainingEpochs++;
            this.draw();
            
            if (this.phaseTrainingEpochs >= this.targetEpochs) {
                this.pauseTrain();
                document.getElementById('train-phase2').disabled = false;
                this.updatePhase(1, 'Phase 1 complete. Ready for Phase 2.');
            }
        }, 100);
    }

    trainPhase2() {
        if (this.isTraining) return;
        
        this.pauseTrain();
        this.currentDataset = this.dataset2;
        this.phaseTrainingEpochs = 0;
        this.updatePhase(2, 'Training on new dataset...');
        
        document.getElementById('train-phase2').disabled = true;
        
        this.isTraining = true;
        this.trainingInterval = setInterval(() => {
            this.classifier.trainStep(this.currentDataset);
            this.phaseTrainingEpochs++;
            this.draw();
            
            if (this.phaseTrainingEpochs >= this.targetEpochs) {
                this.pauseTrain();
                document.getElementById('test-original').disabled = false;
                this.updatePhase(2, 'Phase 2 complete. Ready to test original data.');
            }
        }, 100);
    }

    testOriginal() {
        this.pauseTrain();
        this.updatePhase(3, 'Testing on original data - observe catastrophic forgetting!');
        document.getElementById('test-original').disabled = true;
        this.draw();
    }

    stepTrain() {
        if (this.currentDataset) {
            this.classifier.trainStep(this.currentDataset);
            this.draw();
        }
    }

    autoTrain() {
        if (this.trainingInterval) return;
        
        this.isTraining = true;
        this.trainingInterval = setInterval(() => {
            if (this.currentDataset) {
                this.classifier.trainStep(this.currentDataset);
                this.draw();
            }
        }, 100);
    }

    pauseTrain() {
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
            this.trainingInterval = null;
            this.isTraining = false;
        }
    }

    reset() {
        this.pauseTrain();
        this.classifier.reset();
        this.currentDataset = this.dataset1;
        this.phase = 1;
        this.phaseTrainingEpochs = 0;
        
        document.getElementById('train-phase1').disabled = false;
        document.getElementById('train-phase2').disabled = true;
        document.getElementById('test-original').disabled = true;
        
        this.updatePhase(1, 'Ready to begin training');
        this.draw();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CatastrophicForgettingViz();
});