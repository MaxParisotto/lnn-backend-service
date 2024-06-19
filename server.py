from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models
import redis
import json
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Initialize Redis client
redis_client = redis.Redis(host='10.0.1.35', port=6379, decode_responses=True)

class LiquidLayer(layers.Layer):
    def __init__(self, units=64):
        super(LiquidLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.tanh(tf.matmul(inputs, self.w) + self.b)

def build_liquid_network(input_shape, liquid_units, dense_units, output_units):
    model = models.Sequential()
    model.add(LiquidLayer(liquid_units))  # Liquid Layer with variable units
    model.add(layers.Dense(dense_units, activation='relu'))  # Dense Layer with variable units
    model.add(layers.Dense(output_units))  # Output Layer with variable units
    return model

@app.route('/update-parameters', methods=['POST'])
def update_parameters():
    data = request.get_json()
    liquid_units = data.get('liquid_units', 64)
    dense_units = data.get('dense_units', 32)
    learning_rate = data.get('learning_rate', 0.001)
    
    # Build and compile the model with new parameters
    input_shape = (None, 10)
    output_units = 1
    global model_instance
    model_instance = build_liquid_network(input_shape, liquid_units, dense_units, output_units)
    model_instance.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    return jsonify({"success": True, "status": "parameters updated"})

@app.route('/status', methods=['GET'])
def status():
    config = {
        "liquid_units": model_instance.layers[0].units,
        "dense_units": model_instance.layers[1].units,
        "learning_rate": model_instance.optimizer.learning_rate.numpy(),
        "status": "running"
    }
    return jsonify(config)

@app.route('/neuron-states', methods=['GET'])
def neuron_states():
    if model_instance is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    neuron_states = model_instance.layers[0].w.numpy().tolist()
    redis_client.set('neuron_states', json.dumps(neuron_states))
    
    return jsonify(neuron_states)

@app.route('/neuron-states-plot', methods=['GET'])
def neuron_states_plot():
    neuron_states = json.loads(redis_client.get('neuron_states') or "[]")
    plt.figure()
    plt.plot(neuron_states)
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Value')
    plt.title('Neuron States')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return jsonify(image_base64)

@app.route('/store-interaction', methods=['POST'])
def store_interaction():
    data = request.get_json()
    redis_client.lpush('interactions', json.dumps(data))
    return jsonify({"success": True})

@app.route('/get-interactions', methods=['GET'])
def get_interactions():
    interactions = [json.loads(item) for item in redis_client.lrange('interactions', 0, -1)]
    return jsonify(interactions)

if __name__ == '__main__':
    model_instance = build_liquid_network((None, 10), 64, 32, 1)
    model_instance.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    app.run(host='0.0.0.0', port=5000)