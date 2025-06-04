class MLP4ClassClassifier {
    constructor(learningRate = 0.1, hiddenSize = 8) {
        this.learningRate = learningRate;
        this.hiddenSize = hiddenSize;
        this.numClasses = 4;
        
        // Initialize weights with small random values
        this.weights1 = this.initializeWeights(2, hiddenSize); // Input to hidden
        this.bias1 = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
        
        this.weights2 = this.initializeWeights(hiddenSize, this.numClasses); // Hidden to output (4 classes)
        this.bias2 = new Array(this.numClasses).fill(0).map(() => Math.random() * 0.1 - 0.05);
        
        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.weightHistory = [];
        
        // Store initial weights
        this.saveWeightSnapshot();
    }

    initializeWeights(inputSize, outputSize) {
        const weights = [];
        for (let i = 0; i < inputSize; i++) {
            weights[i] = [];
            for (let j = 0; j < outputSize; j++) {
                weights[i][j] = Math.random() * 0.2 - 0.1;
            }
        }
        return weights;
    }

    saveWeightSnapshot() {
        this.weightHistory.push({
            epoch: this.epoch,
            weights1: JSON.parse(JSON.stringify(this.weights1)),
            bias1: [...this.bias1],
            weights2: JSON.parse(JSON.stringify(this.weights2)),
            bias2: [...this.bias2]
        });
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
    }

    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((sum, exp) => sum + exp, 0);
        return expLogits.map(exp => exp / sumExp);
    }

    forward(x) {
        // Forward pass through network
        const hidden = [];
        
        // Input to hidden layer
        for (let j = 0; j < this.hiddenSize; j++) {
            let sum = this.bias1[j];
            for (let i = 0; i < x.length; i++) {
                sum += x[i] * this.weights1[i][j];
            }
            hidden[j] = this.sigmoid(sum);
        }
        
        // Hidden to output layer (logits)
        const logits = [];
        for (let k = 0; k < this.numClasses; k++) {
            let sum = this.bias2[k];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.weights2[j][k];
            }
            logits[k] = sum;
        }
        
        const output = this.softmax(logits);
        
        return { hidden: hidden, logits: logits, output: output };
    }

    predict(x) {
        return this.forward(x).output;
    }

    predictClass(x) {
        const predictions = this.predict(x);
        return predictions.indexOf(Math.max(...predictions));
    }

    trainStep(dataset) {
        let totalLoss = 0;
        let correct = 0;

        for (let sample = 0; sample < dataset.length; sample++) {
            const [x, y] = dataset[sample];
            
            // Forward pass
            const { hidden, logits, output } = this.forward(x);
            
            // Calculate cross-entropy loss
            const loss = -Math.log(output[y] + 1e-15);
            totalLoss += loss;

            if (this.predictClass(x) === y) {
                correct++;
            }

            // Backward pass
            const outputErrors = [...output];
            outputErrors[y] -= 1; // derivative of cross-entropy + softmax
            
            // Calculate hidden layer errors
            const hiddenErrors = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                let error = 0;
                for (let k = 0; k < this.numClasses; k++) {
                    error += outputErrors[k] * this.weights2[j][k];
                }
                hiddenErrors[j] = error * hidden[j] * (1 - hidden[j]);
            }
            
            // Update output layer weights
            for (let j = 0; j < this.hiddenSize; j++) {
                for (let k = 0; k < this.numClasses; k++) {
                    this.weights2[j][k] -= this.learningRate * outputErrors[k] * hidden[j];
                }
            }
            for (let k = 0; k < this.numClasses; k++) {
                this.bias2[k] -= this.learningRate * outputErrors[k];
            }
            
            // Update hidden layer weights
            for (let i = 0; i < x.length; i++) {
                for (let j = 0; j < this.hiddenSize; j++) {
                    this.weights1[i][j] -= this.learningRate * hiddenErrors[j] * x[i];
                }
            }
            for (let j = 0; j < this.hiddenSize; j++) {
                this.bias1[j] -= this.learningRate * hiddenErrors[j];
            }
        }

        this.epoch++;
        const avgLoss = totalLoss / dataset.length;
        const accuracy = correct / dataset.length;
        
        this.lossHistory.push(avgLoss);
        this.accuracyHistory.push(accuracy);
        
        // Save weight snapshot every few epochs
        if (this.epoch % 5 === 0) {
            this.saveWeightSnapshot();
        }

        return { loss: avgLoss, accuracy: accuracy };
    }

    train(dataset, epochs = 100, onEpochComplete = null) {
        for (let i = 0; i < epochs; i++) {
            const metrics = this.trainStep(dataset);
            if (onEpochComplete) {
                onEpochComplete(metrics, i + 1);
            }
        }
    }

    // Generate decision boundary points for visualization
    getDecisionBoundary(xMin, xMax, yMin, yMax, resolution = 50) {
        const points = [];
        const stepX = (xMax - xMin) / resolution;
        const stepY = (yMax - yMin) / resolution;
        
        for (let x = xMin; x <= xMax; x += stepX) {
            for (let y = yMin; y <= yMax; y += stepY) {
                const predictions = this.predict([x, y]);
                const predictedClass = predictions.indexOf(Math.max(...predictions));
                const confidence = Math.max(...predictions);
                points.push({ x: x, y: y, class: predictedClass, confidence: confidence, predictions: predictions });
            }
        }
        return points;
    }

    reset() {
        this.weights1 = this.initializeWeights(2, this.hiddenSize);
        this.bias1 = new Array(this.hiddenSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
        this.weights2 = this.initializeWeights(this.hiddenSize, this.numClasses);
        this.bias2 = new Array(this.numClasses).fill(0).map(() => Math.random() * 0.1 - 0.05);
        
        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.weightHistory = [];
        
        this.saveWeightSnapshot();
    }

    getMetrics() {
        return {
            epoch: this.epoch,
            loss: this.lossHistory[this.lossHistory.length - 1] || 0,
            accuracy: this.accuracyHistory[this.accuracyHistory.length - 1] || 0,
            weights1: this.weights1,
            bias1: this.bias1,
            weights2: this.weights2,
            bias2: this.bias2,
            weightHistory: this.weightHistory
        };
    }

    getWeightMagnitudes() {
        let w1Mag = 0, w2Mag = 0;
        
        // Calculate L2 norm of weights1
        for (let i = 0; i < this.weights1.length; i++) {
            for (let j = 0; j < this.weights1[i].length; j++) {
                w1Mag += this.weights1[i][j] * this.weights1[i][j];
            }
        }
        w1Mag = Math.sqrt(w1Mag);
        
        // Calculate L2 norm of weights2
        for (let j = 0; j < this.weights2.length; j++) {
            for (let k = 0; k < this.weights2[j].length; k++) {
                w2Mag += this.weights2[j][k] * this.weights2[j][k];
            }
        }
        w2Mag = Math.sqrt(w2Mag);
        
        return { hidden: w1Mag, output: w2Mag };
    }
}

// Task 1: Classes 0 and 1 (first two classes)
function generateTask1Dataset() {
    return [
        // Class 0 (red) - top-left quadrant
        [[1.5, 6.5], 0], [[2, 7], 0], [[1, 6], 0], [[2.5, 6.5], 0],
        [[1.5, 7.5], 0], [[2, 6], 0], [[1, 7], 0], [[2.5, 7.5], 0],
        
        // Class 1 (green) - bottom-right quadrant  
        [[6, 1.5], 1], [[7, 2], 1], [[6.5, 1], 1], [[7.5, 2.5], 1],
        [[6.5, 2.5], 1], [[7, 1.5], 1], [[6, 2], 1], [[7.5, 1.5], 1]
    ];
}

// Task 2: Classes 2 and 3 (second two classes)
function generateTask2Dataset() {
    return [
        // Class 2 (blue) - top-right quadrant
        [[6, 6.5], 2], [[7, 7], 2], [[6.5, 6], 2], [[7.5, 6.5], 2],
        [[6.5, 7.5], 2], [[7, 6], 2], [[6, 7], 2], [[7.5, 7.5], 2],
        
        // Class 3 (yellow) - bottom-left quadrant
        [[1.5, 1.5], 3], [[2, 2], 3], [[1, 1], 3], [[2.5, 2.5], 3],
        [[1.5, 2.5], 3], [[2, 1], 3], [[1, 2], 3], [[2.5, 1.5], 3]
    ];
}

// Combined dataset for joint training comparison
function generateAllClassesDataset() {
    return [...generateTask1Dataset(), ...generateTask2Dataset()];
}

// Get class colors for visualization
function getClassColor(classId, highlight = false) {
    const colors = [
        { normal: '#ff6666', highlight: '#ff4444', stroke: '#cc0000' }, // Class 0: Red
        { normal: '#66ff66', highlight: '#44ff44', stroke: '#00cc00' }, // Class 1: Green  
        { normal: '#6666ff', highlight: '#4444ff', stroke: '#0000cc' }, // Class 2: Blue
        { normal: '#ffff66', highlight: '#ffff44', stroke: '#cccc00' }  // Class 3: Yellow
    ];
    return colors[classId];
}

// Get class name for display
function getClassName(classId) {
    const names = ['Red', 'Green', 'Blue', 'Yellow'];
    return names[classId];
}