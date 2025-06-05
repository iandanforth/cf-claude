class MLPCatastrophicForgettingViz {
    constructor() {
        this.decisionCanvas = document.getElementById('decision-canvas');
        this.decisionCtx = this.decisionCanvas.getContext('2d');
        this.weightCanvas = document.getElementById('weight-canvas');
        this.weightCtx = this.weightCanvas.getContext('2d');
        
        this.classifier = new MLPClassifier(0.1, 4); // Learning rate 0.1, 4 hidden neurons
        
        // Initialize network introspection
        this.networkIntrospection = new NetworkIntrospection(this.weightCanvas, this.classifier);
        this.networkIntrospection.setHoverCallback(() => this.draw());
        
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
    }

    worldToCanvas(worldX, worldY, canvas) {
        const canvasX = ((worldX - this.xMin) / (this.xMax - this.xMin)) * canvas.width;
        const canvasY = canvas.height - ((worldY - this.yMin) / (this.yMax - this.yMin)) * canvas.height;
        return [canvasX, canvasY];
    }

    drawDecisionBackground() {
        const boundary = this.classifier.getDecisionBoundary(this.xMin, this.xMax, this.yMin, this.yMax, 40);
        
        for (const point of boundary) {
            const [canvasX, canvasY] = this.worldToCanvas(point.x, point.y, this.decisionCanvas);
            
            const intensity = Math.abs(point.prediction - 0.5) * 2;
            const alpha = intensity * 0.3;
            
            if (point.prediction > 0.5) {
                this.decisionCtx.fillStyle = `rgba(0, 100, 255, ${alpha})`;
            } else {
                this.decisionCtx.fillStyle = `rgba(255, 100, 0, ${alpha})`;
            }
            
            this.decisionCtx.fillRect(canvasX - 3, canvasY - 3, 6, 6);
        }
    }

    drawDataPoints(dataset, highlight = false) {
        for (let i = 0; i < dataset.length; i++) {
            const [point, label] = dataset[i];
            const [canvasX, canvasY] = this.worldToCanvas(point[0], point[1], this.decisionCanvas);
            
            this.decisionCtx.beginPath();
            this.decisionCtx.arc(canvasX, canvasY, highlight ? 12 : 8, 0, 2 * Math.PI);
            
            if (label === 0) {
                this.decisionCtx.fillStyle = highlight ? '#ff4444' : '#ff6666';
                this.decisionCtx.strokeStyle = '#cc0000';
            } else {
                this.decisionCtx.fillStyle = highlight ? '#4444ff' : '#6666ff';
                this.decisionCtx.strokeStyle = '#0000cc';
            }
            
            this.decisionCtx.fill();
            this.decisionCtx.lineWidth = 2;
            this.decisionCtx.stroke();
            
            if (highlight) {
                const prediction = this.classifier.predictClass(point);
                const isCorrect = prediction === label;
                
                this.decisionCtx.fillStyle = isCorrect ? '#00aa00' : '#aa0000';
                this.decisionCtx.font = 'bold 12px Arial';
                this.decisionCtx.textAlign = 'center';
                this.decisionCtx.textBaseline = 'middle';
                this.decisionCtx.fillText(isCorrect ? '✓' : '✗', canvasX, canvasY);
                this.decisionCtx.textBaseline = 'alphabetic';
            }
        }
    }

    drawAxes() {
        this.decisionCtx.strokeStyle = '#ddd';
        this.decisionCtx.lineWidth = 1;
        
        for (let i = 0; i <= 8; i++) {
            const [x, y1] = this.worldToCanvas(i, this.yMin, this.decisionCanvas);
            const [, y2] = this.worldToCanvas(i, this.yMax, this.decisionCanvas);
            this.decisionCtx.beginPath();
            this.decisionCtx.moveTo(x, y1);
            this.decisionCtx.lineTo(x, y2);
            this.decisionCtx.stroke();
            
            const [x1, y] = this.worldToCanvas(this.xMin, i, this.decisionCanvas);
            const [x2, ] = this.worldToCanvas(this.xMax, i, this.decisionCanvas);
            this.decisionCtx.beginPath();
            this.decisionCtx.moveTo(x1, y);
            this.decisionCtx.lineTo(x2, y);
            this.decisionCtx.stroke();
        }
    }

    drawWeightVisualization() {
        this.weightCtx.clearRect(0, 0, this.weightCanvas.width, this.weightCanvas.height);
        
        const metrics = this.classifier.getMetrics();
        const weights1 = metrics.weights1; // 2x4 matrix (input to hidden)
        const weights2 = metrics.weights2; // 4x1 matrix (hidden to output)
        
        // Draw network architecture
        const inputX = 50;
        const hiddenX = 200;
        const outputX = 350;
        const centerY = this.weightCanvas.height / 2;
        
        // Input layer positions
        const inputPositions = [
            [inputX, centerY - 30],
            [inputX, centerY + 30]
        ];
        
        // Hidden layer positions
        const hiddenPositions = [
            [hiddenX, centerY - 45],
            [hiddenX, centerY - 15],
            [hiddenX, centerY + 15],
            [hiddenX, centerY + 45]
        ];
        
        // Output position
        const outputPosition = [outputX, centerY];
        
        // Draw connections with thickness based on weight magnitude
        const maxWeight = Math.max(
            Math.max(...weights1.flat().map(Math.abs)),
            Math.max(...weights2.flat().map(Math.abs))
        );
        
        // Input to hidden connections
        for (let i = 0; i < inputPositions.length; i++) {
            for (let j = 0; j < hiddenPositions.length; j++) {
                const weight = weights1[i][j];
                const thickness = Math.abs(weight) / maxWeight * 5 + 0.5;
                const color = weight > 0 ? '#2196F3' : '#F44336';
                
                this.weightCtx.strokeStyle = color;
                this.weightCtx.lineWidth = thickness;
                this.weightCtx.beginPath();
                this.weightCtx.moveTo(inputPositions[i][0], inputPositions[i][1]);
                this.weightCtx.lineTo(hiddenPositions[j][0], hiddenPositions[j][1]);
                this.weightCtx.stroke();
            }
        }
        
        // Hidden to output connections
        for (let j = 0; j < hiddenPositions.length; j++) {
            const weight = weights2[j][0];
            const thickness = Math.abs(weight) / maxWeight * 5 + 0.5;
            const color = weight > 0 ? '#2196F3' : '#F44336';
            
            this.weightCtx.strokeStyle = color;
            this.weightCtx.lineWidth = thickness;
            this.weightCtx.beginPath();
            this.weightCtx.moveTo(hiddenPositions[j][0], hiddenPositions[j][1]);
            this.weightCtx.lineTo(outputPosition[0], outputPosition[1]);
            this.weightCtx.stroke();
        }
        
        // Draw neurons
        const drawNeuron = (x, y, label, color = '#4CAF50') => {
            this.weightCtx.beginPath();
            this.weightCtx.arc(x, y, 15, 0, 2 * Math.PI);
            this.weightCtx.fillStyle = color;
            this.weightCtx.fill();
            this.weightCtx.strokeStyle = '#333';
            this.weightCtx.lineWidth = 2;
            this.weightCtx.stroke();
            
            this.weightCtx.fillStyle = 'white';
            this.weightCtx.font = 'bold 10px Arial';
            this.weightCtx.textAlign = 'center';
            this.weightCtx.fillText(label, x, y + 3);
        };
        
        // Draw input neurons
        drawNeuron(inputPositions[0][0], inputPositions[0][1], 'X₁', '#FF9800');
        drawNeuron(inputPositions[1][0], inputPositions[1][1], 'X₂', '#FF9800');
        
        // Draw hidden neurons
        for (let j = 0; j < hiddenPositions.length; j++) {
            drawNeuron(hiddenPositions[j][0], hiddenPositions[j][1], `H${j+1}`, '#4CAF50');
        }
        
        // Draw output neuron
        drawNeuron(outputPosition[0], outputPosition[1], 'Y', '#9C27B0');
        
        // Update introspection with current network positions (15px radius for all neurons)
        this.networkIntrospection.setNetworkPositions(inputPositions, hiddenPositions, 15, 15);
        
        // Draw labels
        this.weightCtx.fillStyle = '#333';
        this.weightCtx.font = '12px Arial';
        this.weightCtx.textAlign = 'center';
        this.weightCtx.fillText('Input', inputX, 20);
        this.weightCtx.fillText('Hidden', hiddenX, 20);
        this.weightCtx.fillText('Output', outputX, 20);
        
        // Update weight magnitude display
        const magnitudes = this.classifier.getWeightMagnitudes();
        document.getElementById('weight-stats').innerHTML = `
            Hidden Layer Magnitude: ${magnitudes.hidden.toFixed(3)}<br>
            Output Layer Magnitude: ${magnitudes.output.toFixed(3)}
        `;
    }

    draw() {
        // Clear decision canvas
        this.decisionCtx.clearRect(0, 0, this.decisionCanvas.width, this.decisionCanvas.height);
        
        this.drawAxes();
        this.drawDecisionBackground();
        
        if (this.phase === 1 || this.phase === 2) {
            this.drawDataPoints(this.currentDataset);
        }
        
        if (this.phase === 3) {
            this.drawDataPoints(this.dataset1, true);
        }
        
        this.drawWeightVisualization();
        
        // Draw network introspection overlay
        this.networkIntrospection.redraw();
        
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
        
        const magnitudes = this.classifier.getWeightMagnitudes();
        
        document.getElementById('epoch-value').textContent = metrics.epoch;
        document.getElementById('accuracy-value').textContent = `${accuracyPercent.toFixed(1)}%`;
        document.getElementById('accuracy-value').className = `metric-value accuracy-value ${accuracyClass}`;
        document.getElementById('loss-value').textContent = metrics.loss.toFixed(3);
        document.getElementById('hidden-weights').textContent = magnitudes.hidden.toFixed(3);
        document.getElementById('output-weights').textContent = magnitudes.output.toFixed(3);
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
    new MLPCatastrophicForgettingViz();
});