class NetworkIntrospection {
    constructor(weightCanvas, classifier) {
        this.weightCanvas = weightCanvas;
        this.weightCtx = weightCanvas.getContext('2d');
        this.classifier = classifier;
        this.hoveredHiddenUnit = null;
        this.hiddenPositions = [];
        this.inputPositions = [];
        this.isHovering = false;
        this.inputRadius = 15;
        this.hiddenRadius = 15;
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.weightCanvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.weightCanvas.addEventListener('mouseleave', () => this.handleMouseLeave());
    }

    setNetworkPositions(inputPositions, hiddenPositions, inputRadius = 15, hiddenRadius = 15) {
        this.inputPositions = inputPositions;
        this.hiddenPositions = hiddenPositions;
        this.inputRadius = inputRadius;
        this.hiddenRadius = hiddenRadius;
    }

    handleMouseMove(e) {
        const rect = this.weightCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let hoveredUnit = null;
        
        // Check if mouse is over any hidden unit
        for (let i = 0; i < this.hiddenPositions.length; i++) {
            const [x, y] = this.hiddenPositions[i];
            const distance = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2);
            
            if (distance <= this.hiddenRadius) { // Use configured radius for hover detection
                hoveredUnit = i;
                break;
            }
        }
        
        if (hoveredUnit !== this.hoveredHiddenUnit) {
            this.hoveredHiddenUnit = hoveredUnit;
            this.isHovering = hoveredUnit !== null;
            
            if (this.onHoverChange) {
                this.onHoverChange(this.hoveredHiddenUnit, this.isHovering);
            }
        }
    }

    handleMouseLeave() {
        if (this.hoveredHiddenUnit !== null) {
            this.hoveredHiddenUnit = null;
            this.isHovering = false;
            
            if (this.onHoverChange) {
                this.onHoverChange(null, false);
            }
        }
    }

    // Helper function to calculate line endpoint at circle edge
    calculateLineEndpoint(startPos, endPos, circleRadius) {
        const dx = endPos[0] - startPos[0];
        const dy = endPos[1] - startPos[1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance === 0) return endPos;
        
        const unitX = dx / distance;
        const unitY = dy / distance;
        
        return [
            endPos[0] - unitX * circleRadius,
            endPos[1] - unitY * circleRadius
        ];
    }

    drawInputWeightHighlight() {
        if (!this.isHovering || this.hoveredHiddenUnit === null) {
            return;
        }

        const metrics = this.classifier.getMetrics();
        const weights1 = metrics.weights1; // Input to hidden weights
        
        if (!weights1 || !this.inputPositions || !this.hiddenPositions) {
            return;
        }

        const hiddenIndex = this.hoveredHiddenUnit;
        const hiddenPos = this.hiddenPositions[hiddenIndex];
        
        // Find max weight magnitude for this hidden unit
        const hiddenWeights = [];
        for (let i = 0; i < this.inputPositions.length; i++) {
            hiddenWeights.push(weights1[i][hiddenIndex]);
        }
        const maxWeight = Math.max(...hiddenWeights.map(Math.abs));
        
        // Draw highlighted connections from inputs to this hidden unit
        for (let i = 0; i < this.inputPositions.length; i++) {
            const weight = weights1[i][hiddenIndex];
            const inputPos = this.inputPositions[i];
            
            // Calculate line endpoints that stop at circle edges
            const lineStart = this.calculateLineEndpoint(hiddenPos, inputPos, this.inputRadius);
            const lineEnd = this.calculateLineEndpoint(inputPos, hiddenPos, this.hiddenRadius);
            
            // Calculate line properties
            const thickness = Math.abs(weight) / maxWeight * 8 + 2; // Thicker for highlight
            const alpha = Math.min(Math.abs(weight) / maxWeight * 0.9 + 0.3, 1);
            const color = weight > 0 ? `rgba(0, 200, 0, ${alpha})` : `rgba(200, 0, 0, ${alpha})`;
            
            // Draw the connection line
            this.weightCtx.strokeStyle = color;
            this.weightCtx.lineWidth = thickness;
            this.weightCtx.beginPath();
            this.weightCtx.moveTo(lineStart[0], lineStart[1]);
            this.weightCtx.lineTo(lineEnd[0], lineEnd[1]);
            this.weightCtx.stroke();
            
            // Draw weight value label at midpoint
            const midX = (lineStart[0] + lineEnd[0]) / 2;
            const midY = (lineStart[1] + lineEnd[1]) / 2;
            
            // Background for weight text
            this.weightCtx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            this.weightCtx.font = 'bold 10px Arial';
            this.weightCtx.textAlign = 'center';
            const text = weight.toFixed(2);
            const textWidth = this.weightCtx.measureText(text).width;
            this.weightCtx.fillRect(midX - textWidth/2 - 2, midY - 6, textWidth + 4, 12);
            
            // Weight text
            this.weightCtx.fillStyle = weight > 0 ? '#006600' : '#660000';
            this.weightCtx.fillText(text, midX, midY + 3);
        }
        
        // Highlight the hovered hidden unit with a ring around it
        this.weightCtx.beginPath();
        this.weightCtx.arc(hiddenPos[0], hiddenPos[1], 18, 0, 2 * Math.PI);
        this.weightCtx.strokeStyle = '#FFD700';
        this.weightCtx.lineWidth = 3;
        this.weightCtx.stroke();
    }

    // Method to be called by visualization classes when redrawing
    redraw() {
        // This will be called after the normal weight visualization is drawn
        // to overlay the introspection highlighting
        this.drawInputWeightHighlight();
    }

    // Set callback for when hover state changes
    setHoverCallback(callback) {
        this.onHoverChange = callback;
    }
}