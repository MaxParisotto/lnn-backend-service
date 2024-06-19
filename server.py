from flask import Flask, request, jsonify
import tensorflow as tf
from ncps.tf import LTCCell
from ncps.wirings import NCPWiring  # Correct import from NCP examples
import redis
import json
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Initialize Redis client
redis_client = redis.Redis(host='10.0.1.35', port=6379, decode_responses=True)

def build_liquid_network(input_shape, liquid_units, dense_units, output_units):
    # Define the wiring for the LTCCell
    wiring = NCPWiring(input_dim=input_shape[1], output_dim=output_units, num_neurons=liquid_units)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(LTCCell(wiring), input_shape=input_shape))  # NCP Liquid Layer with variable units
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))  # Dense Layer with variable units
    model.add(tf.keras.layers.Dense(output_units))  # Output Layer with variable units
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
    model = build_liquid_network(input_shape, liquid_units, dense_units, output_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    global model_instance
    model_instance = model
    
    return jsonify({"success": True, "status": "parameters updated"})

@app.route('/status', methods=['GET'])
def status():
    config = {
        "liquid_units": model_instance.layers[0].cell.units,
        "dense_units": model_instance.layers[1].units,
        "learning_rate": model_instance.optimizer.learning_rate.numpy(),
        "status": "running"
    }
    return jsonify(config)

@app.route('/neuron-states', methods=['GET'])
def neuron_states():
    if model_instance is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    neuron_states = model_instance.layers[0].cell.get_weights()[0].tolist()
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