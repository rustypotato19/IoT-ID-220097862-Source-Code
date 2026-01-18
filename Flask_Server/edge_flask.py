from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from collections import deque
import requests
import math
from datetime import datetime

# ----------------------
# ThingSpeak Settings
# ----------------------
THINGSPEAK_API_KEY = "N0140F40DPB471ZL"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

# ----------------------
# Load model
# ----------------------
interpreter = tflite.Interpreter(model_path="greenhouse_model_lag10.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------
# Load scaler params (expects columns: feature, mean, scale)
# ----------------------
scaler_df = pd.read_csv("scaler_params.csv")
required_cols = {"feature", "mean", "scale"}

if not required_cols.issubset(set(scaler_df.columns)):
    raise ValueError(f"scaler_params.csv must contain columns: {required_cols}. Found: {list(scaler_df.columns)}")

# Use feature names exactly as stored in CSV (capitalised)
feature_order = scaler_df["feature"].astype(str).str.strip().tolist()
means = scaler_df["mean"].astype(float).to_numpy()
scales = scaler_df["scale"].astype(float).to_numpy()

# ----------------------
# Keep last 11 readings to create 10 lags
# ----------------------
lag_steps = 10
temp_history = deque(maxlen=lag_steps + 1)
hum_history = deque(maxlen=lag_steps + 1)

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)


def encode_action(actions):
    if not actions:
        return 0

    action = actions[0]
    mapping = {
        "TURN_ON_FAN": 1,
        "TURN_ON_HEATER": 2,
        "DEHUMIDIFIER_ON": 3,
        "HUMIDIFIER_ON": 4
    }
    return mapping.get(action, 0)


def send_to_thingspeak(current_temp, current_hum, pred_temp, pred_hum, action_code):
    payload = {
        "api_key": THINGSPEAK_API_KEY,
        "field1": float(pred_temp),
        "field2": float(pred_hum),
        "field3": float(current_temp),
        "field4": float(current_hum),
        "field5": int(action_code),
    }

    try:
        requests.get(THINGSPEAK_URL, params=payload, timeout=5)
    except Exception as e:
        print("ThingSpeak upload failed:", e)


def build_feature_vector(temp, hum):
    """
    Builds feature vector matching exactly the names in scaler_params.csv:
      Temp_Avg, Hum_Avg,
      Temp_lag1..Temp_lag10,
      Hum_lag1..Hum_lag10,
      Hour_sin, Hour_cos
    """

    now = datetime.now()
    hour = now.hour + now.minute / 60.0

    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    if len(temp_history) < lag_steps + 1 or len(hum_history) < lag_steps + 1:
        return None, f"Collected {len(temp_history)}/{lag_steps + 1} readings"

    temps = list(temp_history)
    hums = list(hum_history)

    feat = {
        "Temp_Avg": temp,
        "Hum_Avg": hum,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
    }

    for i in range(1, lag_steps + 1):
        feat[f"Temp_lag{i}"] = temps[-(i + 1)]
        feat[f"Hum_lag{i}"] = hums[-(i + 1)]

    missing = [name for name in feature_order if name not in feat]

    if missing:
        raise ValueError(f"Missing features required by scaler_params.csv: {missing}")

    vector = [feat[name] for name in feature_order]
    return vector, None


def scale_vector(vector):
    v = np.asarray(vector, dtype=np.float32)
    m = means.astype(np.float32)
    s = scales.astype(np.float32)

    if v.shape[0] != m.shape[0]:
        raise ValueError(f"Input length {v.shape[0]} does not match scaler length {m.shape[0]}")

    return (v - m) / s


@app.route("/sensor", methods=["GET"])
def sensor_data():
    temp = request.args.get("temp", type=float)
    hum = request.args.get("hum", type=float)

    if temp is None or hum is None:
        return jsonify({"status": "error", "message": "Missing temp or hum query parameter"}), 400

    temp_history.append(temp)
    hum_history.append(hum)

    feature_values, waiting_msg = build_feature_vector(temp, hum)

    if feature_values is None:
        return jsonify({"status": "waiting", "message": waiting_msg}), 200

    input_scaled = scale_vector(feature_values).reshape(1, -1).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_scaled)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    pred_temp, pred_hum = float(output_data[0][0]), float(output_data[0][1])

    actions = []

    MAX_TEMP = 30.0
    MIN_TEMP = 18.0
    MAX_HUM = 80.0
    MIN_HUM = 40.0

    if pred_temp > MAX_TEMP:
        actions.append("TURN_ON_FAN")
    elif pred_temp < MIN_TEMP:
        actions.append("TURN_ON_HEATER")

    if pred_hum > MAX_HUM:
        actions.append("DEHUMIDIFIER_ON")
    elif pred_hum < MIN_HUM:
        actions.append("HUMIDIFIER_ON")

    action_code = encode_action(actions)

    send_to_thingspeak(temp, hum, pred_temp, pred_hum, action_code)

    return jsonify({
        "predicted_temp": pred_temp,
        "predicted_hum": pred_hum,
        "actions": actions
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)