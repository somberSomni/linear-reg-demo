const model = tf.sequential();
//units refer to the number of nodes in your layer
//we specify the inputshape for the input layer

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});
//tensorflox infers number of inputs coming from
//hidden layer
const output = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
});
model.add(hidden);
model.add(output);
//stochastic gradient descent
const optimizer = tf.train.sgd(0.1);
model.compile({ optimizer, loss: 'meanSquaredError'});

