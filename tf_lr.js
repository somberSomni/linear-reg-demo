//graphical linear regression

window.onload = function () {
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    let xs = [];
    let ys = [];
    let dot_ys = [];
    let dot_xs = [];
    //create the slobe and y-intercept for linear regression
    m = tf.scalar(Math.random()).variable();
    b = tf.scalar(Math.random()).variable();

    const learning_rate = 0.1;
    //using tensorflos stochastic gradrient descent
    const optimizer = tf.train.sgd(learning_rate);
    function predict(x) {
        //y = mx + b
        return x.mul(m).add(b);
    }
    function loss(pred, labels) {
        //chaning loss function y= 1/m * (pred - ys)^2
        return pred.sub(labels).square().mean();
    }
    function mean(arr) {
        return arr.reduce((pV, cV) => pV + cV)/ arr.length
    }
    function meanNorm(n) {
        return (n - canvas.width/2) / canvas.width;
    }
    function reverseNorm(n) {
        return n * canvas.width + canvas.width/2;
    }
    document.getElementById('screen-info').onclick = function (e) {
        const linContainer = document.getElementById('linreg');
        linContainer.removeChild(linContainer.children[0]);
        const rect = canvas.getBoundingClientRect();
        dot_xs.push(e.clientX - rect.left);
        dot_ys.push(e.clientY - rect.top);
        xs.push(meanNorm(e.clientX - rect.left));
        ys.push(meanNorm(canvas.height - e.clientY + rect.top));
        document.getElementById('reset').setAttribute('class', 'showtwo');
    }

    document.getElementById('reset').onclick = function (e) {
        dot_xs = [];
        dot_ys = [];
        xs = [];
        ys = [];
    }

    canvas.onclick = function (e) {
        const coeff = m;
        const bias = b;

        let coeffEl = document.getElementById('coeff');
        let biasEl = document.getElementById('bias');

        coeffEl.innerHTML = coeff;
        biasEl.innerHTML = bias;
        const rect = canvas.getBoundingClientRect();
        dot_xs.push(e.clientX - rect.left);
        dot_ys.push(e.clientY - rect.top);
        xs.push(meanNorm(e.clientX - rect.left));
        ys.push(meanNorm(canvas.height - e.clientY + rect.top));
    }

    function animate() {
        window.requestAnimationFrame(animate);
        if (xs.length > 0) {
            tf.tidy(() => {
                const tensor_ys = tf.tensor1d(ys);
                const tensor_xs = tf.tensor1d(xs);
                optimizer.minimize(() => loss(predict(tensor_xs), tensor_ys));
            });
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.restore();
            for (var i = 0; i <= xs.length - 1; i++) {
                ctx.beginPath();
                ctx.arc(dot_xs[i], dot_ys[i], 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'white';
                ctx.fill();
            }
            const line_xs = [-0.5, 0.5];
            const tfLine_xs = tf.tensor1d(line_xs);
            const tfLine_ys = tf.tidy(() => predict(tfLine_xs));
            const line_ys = tfLine_ys.dataSync();
            tfLine_xs.dispose();
            tfLine_ys.dispose();
            ctx.beginPath();
            ctx.moveTo(reverseNorm(line_xs[0]), canvas.height - reverseNorm(line_ys[0]));
            ctx.lineTo(reverseNorm(line_xs[1]), canvas.height - reverseNorm(line_ys[1]));
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.save();
            //console.log(tf.memory().numTensors);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    animate();
    window.requestAnimationFrame(animate);
}