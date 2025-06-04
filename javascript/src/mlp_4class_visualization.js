class MLP4ClassCatastrophicForgettingViz {
    constructor() {
        this.decisionCanvas = document.getElementById('decision-canvas');
        this.decisionCtx = this.decisionCanvas.getContext('2d');
        this.weightCanvas = document.getElementById('weight-canvas');
        this.weightCtx = this.weightCanvas.getContext('2d');
        
        this.classifier = new MLP4ClassClassifier(0.15, 8); // Learning rate 0.15, 8 hidden neurons
        
        this.xMin = 0;
        this.xMax = 8;
        this.yMin = 0;
        this.yMax = 8;
        
        this.task1Dataset = generateTask1Dataset(); // Classes 0, 1
        this.task2Dataset = generateTask2Dataset(); // Classes 2, 3
        this.currentDataset = this.task1Dataset;
        
        this.phase = 1; // 1: task1, 2: task2, 3: testing task1
        this.isTraining = false;
        this.trainingInterval = null;
        this.phaseTrainingEpochs = 0;
        this.targetEpochs = 60;
        
        this.initializeControls();
        this.draw();
    }

    initializeControls() {
        document.getElementById('train-task1').addEventListener('click', () => this.trainTask1());
        document.getElementById('train-task2').addEventListener('click', () => this.trainTask2());
        document.getElementById('test-task1').addEventListener('click', () => this.testTask1());
        document.getElementById('reset').addEventListener('click', () => this.reset());
    }

    worldToCanvas(worldX, worldY, canvas) {
        const canvasX = ((worldX - this.xMin) / (this.xMax - this.xMin)) * canvas.width;
        const canvasY = canvas.height - ((worldY - this.yMin) / (this.yMax - this.yMin)) * canvas.height;
        return [canvasX, canvasY];
    }

    drawDecisionBackground() {
        const boundary = this.classifier.getDecisionBoundary(this.xMin, this.xMax, this.yMin, this.yMax, 60);
        
        for (const point of boundary) {
            const [canvasX, canvasY] = this.worldToCanvas(point.x, point.y, this.decisionCanvas);
            
            // Use the predicted class to color the background
            const classColor = getClassColor(point.class);
            const alpha = point.confidence * 0.4; // Use confidence for intensity
            
            this.decisionCtx.fillStyle = `${classColor.normal}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
            this.decisionCtx.fillRect(canvasX - 2, canvasY - 2, 4, 4);
        }
    }

    drawDataPoints(dataset, highlight = false) {
        for (let i = 0; i < dataset.length; i++) {
            const [point, label] = dataset[i];
            const [canvasX, canvasY] = this.worldToCanvas(point[0], point[1], this.decisionCanvas);
            
            const colors = getClassColor(label, highlight);
            
            this.decisionCtx.beginPath();
            this.decisionCtx.arc(canvasX, canvasY, highlight ? 12 : 10, 0, 2 * Math.PI);
            
            this.decisionCtx.fillStyle = highlight ? colors.highlight : colors.normal;
            this.decisionCtx.strokeStyle = colors.stroke;
            this.decisionCtx.fill();
            this.decisionCtx.lineWidth = 2;
            this.decisionCtx.stroke();
            
            if (highlight) {
                const prediction = this.classifier.predictClass(point);
                const isCorrect = prediction === label;
                
                this.decisionCtx.fillStyle = isCorrect ? '#00aa00' : '#aa0000';
                this.decisionCtx.font = 'bold 14px Arial';
                this.decisionCtx.textAlign = 'center';
                this.decisionCtx.textBaseline = 'middle';
                this.decisionCtx.fillText(isCorrect ? '✓' : '✗', canvasX, canvasY);
                this.decisionCtx.textBaseline = 'alphabetic';
                
                // Show predicted class vs actual
                if (!isCorrect) {
                    this.decisionCtx.fillStyle = '#000000';
                    this.decisionCtx.font = '10px Arial';
                    this.decisionCtx.fillText(`→${prediction}`, canvasX + 15, canvasY);
                }
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
        const weights1 = metrics.weights1; // 2x8 matrix (input to hidden)
        const weights2 = metrics.weights2; // 8x4 matrix (hidden to output)
        
        // Network architecture positions
        const inputX = 60;
        const hiddenX = 250;
        const outputX = 440;
        const centerY = this.weightCanvas.height / 2;
        
        // Input layer positions
        const inputPositions = [
            [inputX, centerY - 20],
            [inputX, centerY + 20]
        ];
        
        // Hidden layer positions (8 neurons in 2 columns)
        const hiddenPositions = [
            [hiddenX - 20, centerY - 60], [hiddenX + 20, centerY - 60],
            [hiddenX - 20, centerY - 20], [hiddenX + 20, centerY - 20],
            [hiddenX - 20, centerY + 20], [hiddenX + 20, centerY + 20],
            [hiddenX - 20, centerY + 60], [hiddenX + 20, centerY + 60]
        ];
        
        // Output positions (4 classes)
        const outputPositions = [
            [outputX, centerY - 45],
            [outputX, centerY - 15],
            [outputX, centerY + 15],
            [outputX, centerY + 45]
        ];
        
        // Find max weight for scaling
        const maxWeight = Math.max(
            Math.max(...weights1.flat().map(Math.abs)),
            Math.max(...weights2.flat().map(Math.abs))
        );
        
        // Draw input to hidden connections
        for (let i = 0; i < inputPositions.length; i++) {
            for (let j = 0; j < hiddenPositions.length; j++) {
                const weight = weights1[i][j];
                const thickness = Math.abs(weight) / maxWeight * 4 + 0.5;
                const alpha = Math.min(Math.abs(weight) / maxWeight * 0.8 + 0.2, 1);
                const color = weight > 0 ? `rgba(33, 150, 243, ${alpha})` : `rgba(244, 67, 54, ${alpha})`;
                
                this.weightCtx.strokeStyle = color;
                this.weightCtx.lineWidth = thickness;
                this.weightCtx.beginPath();
                this.weightCtx.moveTo(inputPositions[i][0], inputPositions[i][1]);
                this.weightCtx.lineTo(hiddenPositions[j][0], hiddenPositions[j][1]);
                this.weightCtx.stroke();
            }
        }
        
        // Draw hidden to output connections
        for (let j = 0; j < hiddenPositions.length; j++) {
            for (let k = 0; k < outputPositions.length; k++) {
                const weight = weights2[j][k];
                const thickness = Math.abs(weight) / maxWeight * 4 + 0.5;
                const alpha = Math.min(Math.abs(weight) / maxWeight * 0.8 + 0.2, 1);
                const color = weight > 0 ? `rgba(33, 150, 243, ${alpha})` : `rgba(244, 67, 54, ${alpha})`;
                
                this.weightCtx.strokeStyle = color;
                this.weightCtx.lineWidth = thickness;
                this.weightCtx.beginPath();
                this.weightCtx.moveTo(hiddenPositions[j][0], hiddenPositions[j][1]);
                this.weightCtx.lineTo(outputPositions[k][0], outputPositions[k][1]);
                this.weightCtx.stroke();
            }
        }
        
        // Draw neurons
        const drawNeuron = (x, y, label, color = '#4CAF50', size = 12) => {
            this.weightCtx.beginPath();
            this.weightCtx.arc(x, y, size, 0, 2 * Math.PI);
            this.weightCtx.fillStyle = color;
            this.weightCtx.fill();
            this.weightCtx.strokeStyle = '#333';
            this.weightCtx.lineWidth = 2;
            this.weightCtx.stroke();
            
            this.weightCtx.fillStyle = 'white';
            this.weightCtx.font = 'bold 9px Arial';
            this.weightCtx.textAlign = 'center';
            this.weightCtx.fillText(label, x, y + 3);
        };
        
        // Draw input neurons
        drawNeuron(inputPositions[0][0], inputPositions[0][1], 'X₁', '#FF9800');
        drawNeuron(inputPositions[1][0], inputPositions[1][1], 'X₂', '#FF9800');
        
        // Draw hidden neurons
        for (let j = 0; j < hiddenPositions.length; j++) {
            drawNeuron(hiddenPositions[j][0], hiddenPositions[j][1], `H${j+1}`, '#4CAF50', 10);
        }
        
        // Draw output neurons with class colors
        const outputColors = ['#ff6666', '#66ff66', '#6666ff', '#ffff66'];
        for (let k = 0; k < outputPositions.length; k++) {
            drawNeuron(outputPositions[k][0], outputPositions[k][1], `C${k}`, outputColors[k]);
        }
        
        // Draw labels
        this.weightCtx.fillStyle = '#333';
        this.weightCtx.font = '12px Arial';
        this.weightCtx.textAlign = 'center';
        this.weightCtx.fillText('Input', inputX, 25);
        this.weightCtx.fillText('Hidden (8)', hiddenX, 25);
        this.weightCtx.fillText('Output (4)', outputX, 25);
        
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
            // Show Task 1 data with classification results
            this.drawDataPoints(this.task1Dataset, true);
        }
        
        this.drawWeightVisualization();
        this.updateMetrics();
    }

    updateMetrics() {
        const metrics = this.classifier.getMetrics();
        let accuracy = metrics.accuracy;
        
        // Calculate accuracy for testing phase (Task 1 performance)
        if (this.phase === 3) {
            let correct = 0;
            for (const [point, label] of this.task1Dataset) {
                if (this.classifier.predictClass(point) === label) {
                    correct++;
                }
            }
            accuracy = correct / this.task1Dataset.length;
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
        const taskDesc = document.getElementById('task-description');
        
        status.textContent = message;
        
        if (phase === 1) {
            indicator.textContent = 'Phase 1: Learning Task 1';
            indicator.className = 'phase-indicator phase-1';
            taskDesc.innerHTML = '<strong>Task 1:</strong> Learn to classify Red (Class 0) and Green (Class 1) points';
        } else if (phase === 2) {
            indicator.textContent = 'Phase 2: Learning Task 2';
            indicator.className = 'phase-indicator phase-2';
            taskDesc.innerHTML = '<strong>Task 2:</strong> Learn to classify Blue (Class 2) and Yellow (Class 3) points';
        } else if (phase === 3) {
            indicator.textContent = 'Phase 3: Testing Task 1 Knowledge';
            indicator.className = 'phase-indicator phase-3';
            taskDesc.innerHTML = '<strong>Test:</strong> Can the network still classify Red and Green points?';
        }
    }

    trainTask1() {
        if (this.isTraining) return;
        
        this.pauseTrain();
        this.currentDataset = this.task1Dataset;
        this.phaseTrainingEpochs = 0;
        this.updatePhase(1, 'Training on Task 1 (Classes 0 & 1)...');
        
        document.getElementById('train-task1').disabled = true;
        
        this.isTraining = true;
        this.trainingInterval = setInterval(() => {
            this.classifier.trainStep(this.currentDataset);
            this.phaseTrainingEpochs++;
            this.draw();
            
            if (this.phaseTrainingEpochs >= this.targetEpochs) {
                this.pauseTrain();
                document.getElementById('train-task2').disabled = false;
                this.updatePhase(1, 'Task 1 learning complete. Ready for Task 2.');
            }
        }, 80);
    }

    trainTask2() {
        if (this.isTraining) return;
        
        this.pauseTrain();
        this.currentDataset = this.task2Dataset;
        this.phaseTrainingEpochs = 0;
        this.updatePhase(2, 'Training on Task 2 (Classes 2 & 3)...');
        
        document.getElementById('train-task2').disabled = true;
        
        this.isTraining = true;
        this.trainingInterval = setInterval(() => {
            this.classifier.trainStep(this.currentDataset);
            this.phaseTrainingEpochs++;
            this.draw();
            
            if (this.phaseTrainingEpochs >= this.targetEpochs) {
                this.pauseTrain();
                document.getElementById('test-task1').disabled = false;
                this.updatePhase(2, 'Task 2 learning complete. Ready to test Task 1 knowledge.');
            }
        }, 80);
    }

    testTask1() {
        this.pauseTrain();
        this.updatePhase(3, 'Testing Task 1 knowledge - observe catastrophic forgetting!');
        document.getElementById('test-task1').disabled = true;
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
        this.currentDataset = this.task1Dataset;
        this.phase = 1;
        this.phaseTrainingEpochs = 0;
        
        document.getElementById('train-task1').disabled = false;
        document.getElementById('train-task2').disabled = true;
        document.getElementById('test-task1').disabled = true;
        
        this.updatePhase(1, 'Ready to begin training on Task 1');
        this.draw();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MLP4ClassCatastrophicForgettingViz();
});