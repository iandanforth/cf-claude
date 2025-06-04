class MLPClassifier {
    constructor(learningRate = 0.1, hiddenSize = 4) {
        this.learningRate = learningRate;
        this.hiddenSize = hiddenSize;
        
        // Initialize weights with small random values
        this.weights1 = this.initializeWeights(2, hiddenSize); // Input to hidden
        this.bias1 = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
        
        this.weights2 = this.initializeWeights(hiddenSize, 1); // Hidden to output
        this.bias2 = [Math.random() * 0.1 - 0.05];
        
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
        
        // Hidden to output layer
        let output = this.bias2[0];
        for (let j = 0; j < this.hiddenSize; j++) {
            output += hidden[j] * this.weights2[j][0];
        }
        output = this.sigmoid(output);
        
        return { hidden: hidden, output: output };
    }

    predict(x) {
        return this.forward(x).output;
    }

    predictClass(x) {
        return this.predict(x) >= 0.5 ? 1 : 0;
    }

    trainStep(dataset) {
        let totalLoss = 0;
        let correct = 0;

        for (let sample = 0; sample < dataset.length; sample++) {
            const [x, y] = dataset[sample];
            
            // Forward pass
            const { hidden, output } = this.forward(x);
            
            // Calculate loss
            const loss = -(y * Math.log(output + 1e-15) + (1 - y) * Math.log(1 - output + 1e-15));
            totalLoss += loss;

            if (this.predictClass(x) === y) {
                correct++;
            }

            // Backward pass
            const outputError = output - y;
            
            // Calculate hidden layer errors
            const hiddenErrors = [];
            for (let j = 0; j < this.hiddenSize; j++) {
                hiddenErrors[j] = outputError * this.weights2[j][0] * hidden[j] * (1 - hidden[j]);
            }
            
            // Update output layer weights
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weights2[j][0] -= this.learningRate * outputError * hidden[j];
            }
            this.bias2[0] -= this.learningRate * outputError;
            
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
                const prediction = this.predict([x, y]);
                points.push({ x: x, y: y, prediction: prediction });
            }
        }
        return points;
    }

    reset() {
        this.weights1 = this.initializeWeights(2, this.hiddenSize);
        this.bias1 = new Array(this.hiddenSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
        this.weights2 = this.initializeWeights(this.hiddenSize, 1);
        this.bias2 = [Math.random() * 0.1 - 0.05];
        
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
            w2Mag += this.weights2[j][0] * this.weights2[j][0];
        }
        w2Mag = Math.sqrt(w2Mag);
        
        return { hidden: w1Mag, output: w2Mag };
    }
}

// Use the same datasets as linear classifier for consistency
function generateDataset1() {
    return [
        [[1, 3], 0],    // Class 0 (red)
        [[2, 4], 0],    // Class 0 (red)
        [[1.5, 2.5], 0], // Class 0 (red)
        [[2.5, 3.5], 0], // Class 0 (red)
        [[6, 2], 1],    // Class 1 (blue)
        [[7, 3], 1],    // Class 1 (blue)
        [[6.5, 1.5], 1], // Class 1 (blue)
        [[7.5, 2.5], 1]  // Class 1 (blue)
    ];
}

function generateDataset2() {
    return [
        [[2, 1], 0],    // Class 0 (red) - different region
        [[3, 2], 0],    // Class 0 (red)
        [[2.5, 0.5], 0], // Class 0 (red)
        [[3.5, 1.5], 0], // Class 0 (red)
        [[6, 6], 1],    // Class 1 (blue) - different region
        [[7, 7], 1],    // Class 1 (blue)
        [[5.5, 6.5], 1], // Class 1 (blue)
        [[6.5, 7.5], 1]  // Class 1 (blue)
    ];
}