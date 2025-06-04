// Shared utilities for catastrophic forgetting visualizations

class VizUtils {
    constructor(canvas, xMin = 0, xMax = 8, yMin = 0, yMax = 8) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
    }

    worldToCanvas(worldX, worldY) {
        const canvasX = ((worldX - this.xMin) / (this.xMax - this.xMin)) * this.canvas.width;
        const canvasY = this.canvas.height - ((worldY - this.yMin) / (this.yMax - this.yMin)) * this.canvas.height;
        return [canvasX, canvasY];
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

    drawDataPoint(point, label, classifier, highlight = false, getColorFunc = null) {
        const [canvasX, canvasY] = this.worldToCanvas(point[0], point[1]);
        
        // Default 2-class colors if no color function provided
        const defaultColors = {
            0: { normal: '#ff6666', highlight: '#ff4444', stroke: '#cc0000' },
            1: { normal: '#6666ff', highlight: '#4444ff', stroke: '#0000cc' }
        };
        
        const colors = getColorFunc ? getColorFunc(label, highlight) : defaultColors[label];
        
        this.ctx.beginPath();
        this.ctx.arc(canvasX, canvasY, highlight ? 12 : 8, 0, 2 * Math.PI);
        
        this.ctx.fillStyle = highlight ? colors.highlight : colors.normal;
        this.ctx.strokeStyle = colors.stroke;
        this.ctx.fill();
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        if (highlight) {
            const prediction = classifier.predictClass(point);
            const isCorrect = prediction === label;
            
            this.ctx.fillStyle = isCorrect ? '#00aa00' : '#aa0000';
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(isCorrect ? '✓' : '✗', canvasX, canvasY);
            
            // Reset text baseline to default
            this.ctx.textBaseline = 'alphabetic';
        }
    }

    drawDataPoints(dataset, classifier, highlight = false, getColorFunc = null) {
        for (let i = 0; i < dataset.length; i++) {
            const [point, label] = dataset[i];
            this.drawDataPoint(point, label, classifier, highlight, getColorFunc);
        }
    }

    drawBinaryDecisionBackground(classifier, resolution = 50) {
        const stepX = (this.xMax - this.xMin) / resolution;
        const stepY = (this.yMax - this.yMin) / resolution;
        
        for (let x = this.xMin; x <= this.xMax; x += stepX) {
            for (let y = this.yMin; y <= this.yMax; y += stepY) {
                const prediction = classifier.predict([x, y]);
                const [canvasX, canvasY] = this.worldToCanvas(x, y);
                
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
    }

    drawMultiClassDecisionBackground(classifier, resolution = 50, getColorFunc = null) {
        const stepX = (this.xMax - this.xMin) / resolution;
        const stepY = (this.yMax - this.yMin) / resolution;
        
        for (let x = this.xMin; x <= this.xMax; x += stepX) {
            for (let y = this.yMin; y <= this.yMax; y += stepY) {
                const predictions = classifier.predict([x, y]);
                const predictedClass = predictions.indexOf ? 
                    predictions.indexOf(Math.max(...predictions)) : 
                    (predictions > 0.5 ? 1 : 0);
                const confidence = predictions.indexOf ? 
                    Math.max(...predictions) : 
                    Math.abs(predictions - 0.5) * 2;
                
                const [canvasX, canvasY] = this.worldToCanvas(x, y);
                
                let color;
                if (getColorFunc) {
                    const colors = getColorFunc(predictedClass);
                    color = colors.normal;
                } else {
                    // Default binary colors
                    color = predictedClass > 0.5 ? '#0064ff' : '#ff6400';
                }
                
                const alpha = confidence * 0.4;
                this.ctx.fillStyle = `${color}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
                this.ctx.fillRect(canvasX - 2, canvasY - 2, 4, 4);
            }
        }
    }

    updateMetricsDisplay(classifier, phase, dataset1 = null, elemIds = {}) {
        const metrics = classifier.getMetrics();
        let accuracy = metrics.accuracy;
        
        // Calculate accuracy for testing phase if dataset1 provided
        if (phase === 3 && dataset1) {
            let correct = 0;
            for (const [point, label] of dataset1) {
                if (classifier.predictClass(point) === label) {
                    correct++;
                }
            }
            accuracy = correct / dataset1.length;
        }
        
        const accuracyPercent = accuracy * 100;
        let accuracyClass = 'accuracy-low';
        if (accuracyPercent >= 80) {
            accuracyClass = 'accuracy-high';
        } else if (accuracyPercent >= 50) {
            accuracyClass = 'accuracy-medium';
        }
        
        // Update DOM elements if IDs provided
        if (elemIds.epoch) {
            document.getElementById(elemIds.epoch).textContent = metrics.epoch;
        }
        if (elemIds.accuracy) {
            const accuracyElem = document.getElementById(elemIds.accuracy);
            accuracyElem.textContent = `${accuracyPercent.toFixed(1)}%`;
            accuracyElem.className = `metric-value accuracy-value ${accuracyClass}`;
        }
        if (elemIds.loss) {
            document.getElementById(elemIds.loss).textContent = metrics.loss.toFixed(3);
        }
        
        // Weight magnitudes if classifier supports it
        if (classifier.getWeightMagnitudes && elemIds.hiddenWeights && elemIds.outputWeights) {
            const magnitudes = classifier.getWeightMagnitudes();
            document.getElementById(elemIds.hiddenWeights).textContent = magnitudes.hidden.toFixed(3);
            document.getElementById(elemIds.outputWeights).textContent = magnitudes.output.toFixed(3);
        }
    }

    updatePhaseIndicator(phase, elemIds = {}) {
        const indicator = document.getElementById(elemIds.indicator);
        const status = document.getElementById(elemIds.status);
        const taskDesc = document.getElementById(elemIds.taskDesc);
        
        if (phase === 1) {
            if (indicator) {
                indicator.textContent = 'Phase 1: Training on Dataset 1';
                indicator.className = 'phase-indicator phase-1';
            }
        } else if (phase === 2) {
            if (indicator) {
                indicator.textContent = 'Phase 2: Training on Dataset 2';
                indicator.className = 'phase-indicator phase-2';
            }
        } else if (phase === 3) {
            if (indicator) {
                indicator.textContent = 'Phase 3: Testing Original Data';
                indicator.className = 'phase-indicator phase-3';
            }
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Color functions for different visualization types
const ColorSchemes = {
    binary: {
        getColor: (classId, highlight = false) => {
            const colors = {
                0: { normal: '#ff6666', highlight: '#ff4444', stroke: '#cc0000' },
                1: { normal: '#6666ff', highlight: '#4444ff', stroke: '#0000cc' }
            };
            return colors[classId];
        }
    },
    
    fourClass: {
        getColor: (classId, highlight = false) => {
            const colors = {
                0: { normal: '#ff6666', highlight: '#ff4444', stroke: '#cc0000' }, // Red
                1: { normal: '#66ff66', highlight: '#44ff44', stroke: '#00cc00' }, // Green
                2: { normal: '#6666ff', highlight: '#4444ff', stroke: '#0000cc' }, // Blue
                3: { normal: '#ffff66', highlight: '#ffff44', stroke: '#cccc00' }  // Yellow
            };
            return colors[classId];
        }
    }
};