class LinearClassifier {
    constructor(learningRate = 0.01) {
        this.weights = [Math.random() * 0.1 - 0.05, Math.random() * 0.1 - 0.05];
        this.bias = Math.random() * 0.1 - 0.05;
        this.learningRate = learningRate;
        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
    }

    predict(x) {
        const z = this.weights[0] * x[0] + this.weights[1] * x[1] + this.bias;
        return this.sigmoid(z);
    }

    predictClass(x) {
        return this.predict(x) >= 0.5 ? 1 : 0;
    }

    trainStep(dataset) {
        let totalLoss = 0;
        let correct = 0;

        for (let i = 0; i < dataset.length; i++) {
            const [x, y] = dataset[i];
            
            const prediction = this.predict(x);
            const loss = -(y * Math.log(prediction + 1e-15) + (1 - y) * Math.log(1 - prediction + 1e-15));
            totalLoss += loss;

            if (this.predictClass(x) === y) {
                correct++;
            }

            const error = prediction - y;
            
            this.weights[0] -= this.learningRate * error * x[0];
            this.weights[1] -= this.learningRate * error * x[1];
            this.bias -= this.learningRate * error;
        }

        this.epoch++;
        const avgLoss = totalLoss / dataset.length;
        const accuracy = correct / dataset.length;
        
        this.lossHistory.push(avgLoss);
        this.accuracyHistory.push(accuracy);

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

    getDecisionBoundary(xMin, xMax, yMin, yMax) {
        if (Math.abs(this.weights[1]) < 1e-10) {
            const x = -this.bias / this.weights[0];
            return { type: 'vertical', x: x };
        }
        
        const y1 = -(this.weights[0] * xMin + this.bias) / this.weights[1];
        const y2 = -(this.weights[0] * xMax + this.bias) / this.weights[1];
        
        return { 
            type: 'line', 
            points: [[xMin, y1], [xMax, y2]]
        };
    }

    reset() {
        this.weights = [Math.random() * 0.1 - 0.05, Math.random() * 0.1 - 0.05];
        this.bias = Math.random() * 0.1 - 0.05;
        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
    }

    getMetrics() {
        return {
            epoch: this.epoch,
            loss: this.lossHistory[this.lossHistory.length - 1] || 0,
            accuracy: this.accuracyHistory[this.accuracyHistory.length - 1] || 0,
            weights: [...this.weights],
            bias: this.bias
        };
    }
}

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

function generateTestPoints(xMin, xMax, yMin, yMax, density = 20) {
    const points = [];
    const stepX = (xMax - xMin) / density;
    const stepY = (yMax - yMin) / density;
    
    for (let x = xMin; x <= xMax; x += stepX) {
        for (let y = yMin; y <= yMax; y += stepY) {
            points.push([x, y]);
        }
    }
    return points;
}