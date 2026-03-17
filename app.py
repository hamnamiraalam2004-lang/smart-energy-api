from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load all models ──────────────────────────────────────────
print("Loading models...")

model1 = joblib.load('model1_bill.pkl')
model2 = joblib.load('model2_recommendation.pkl')
model3 = joblib.load('model3_anomaly.pkl')
model4 = joblib.load('model4_device.pkl')
model5 = joblib.load('model5_saving.pkl')

le_rec    = joblib.load('model2_le_rec.pkl')
scaler3   = joblib.load('model3_scaler.pkl')
scaler4   = joblib.load('model4_scaler.pkl')
le_device = joblib.load('model4_le_device.pkl')

print("✅ All models loaded!")

# ── Encoding maps ────────────────────────────────────────────
SEASON_MAP  = {'Summer':3, 'Winter':3, 'Spring':2, 'Autumn':0}
CITY_MAP    = {'Faisalabad':0,'Islamabad':1,'Karachi':2,'Lahore':3,'Multan':4,'Peshawar':5,'Quetta':6}
CONN_MAP    = {'Commercial':0, 'Domestic':1}
AC_TYPE_MAP = {'Inverter':1, 'Non-Inverter':2, 'None':3}
FRIDGE_MAP  = {'Large':0, 'Medium':1, 'Small':2}
SLOT_MAP    = {'normal':0, 'off_peak':1, 'peak':2}

# ── Health check ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'Smart Energy Monitor API',
        'models': ['bill_prediction','recommendation','anomaly_detection','device_detection','peak_saving']
    })

# ── MODEL 1: Bill Prediction ─────────────────────────────────
@app.route('/predict/bill', methods=['POST'])
def predict_bill():
    try:
        data = request.json
        features = [[
            data.get('led_bulbs_count', 0),
            data.get('bulb_hrs_day', 0),
            data.get('fans_count', 0),
            data.get('fan_hrs_day', 0),
            data.get('ac_count', 0),
            data.get('ac_hrs_offpeak', 0),
            data.get('ac_hrs_normal', 0),
            data.get('ac_hrs_peak', 0),
            data.get('fridge_count', 0),
            data.get('geyser_hrs_day', 0),
            data.get('washing_machine_hrs', 0),
            data.get('tv_hrs', 0),
            data.get('misc_watts', 0),
            SEASON_MAP.get(data.get('season', 'Summer'), 3),
            CITY_MAP.get(data.get('city', 'Lahore'), 3),
            CONN_MAP.get(data.get('connection_type', 'Domestic'), 1),
            AC_TYPE_MAP.get(data.get('ac_type', 'None'), 3),
            FRIDGE_MAP.get(data.get('fridge_type', 'Medium'), 1),
        ]]
        prediction = model1.predict(features)[0]
        daily_bill = round(prediction / 30, 1)
        return jsonify({
            'success': True,
            'monthly_bill_rs': round(prediction, 0),
            'daily_bill_rs': daily_bill,
            'annual_bill_rs': round(prediction * 12, 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── MODEL 2: Recommendation ──────────────────────────────────
@app.route('/predict/recommendation', methods=['POST'])
def predict_recommendation():
    try:
        data = request.json
        features = [[
            data.get('peak_cost_fraction', 0),
            data.get('daily_kwh', 0),
            data.get('ac_hrs_peak', 0),
            data.get('geyser_hrs_day', 0),
            data.get('washing_machine_hrs', 0),
            SEASON_MAP.get(data.get('season', 'Summer'), 3),
            CONN_MAP.get(data.get('connection_type', 'Domestic'), 1),
        ]]
        pred_enc = model2.predict(features)[0]
        recommendation = le_rec.inverse_transform([pred_enc])[0]
        proba = model2.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 1)
        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── MODEL 3: Anomaly Detection ───────────────────────────────
@app.route('/predict/anomaly', methods=['POST'])
def predict_anomaly():
    try:
        data = request.json
        features = [[
            data.get('voltage_v', 220),
            data.get('current_a', 0),
            data.get('power_w', 0),
            data.get('expected_min_w', 0),
            data.get('expected_max_w', 100),
            data.get('hour_of_day', 12),
            SLOT_MAP.get(data.get('time_slot', 'normal'), 0),
        ]]
        features_sc = scaler3.transform(features)
        prediction  = model3.predict(features_sc)[0]
        proba       = model3.predict_proba(features_sc)[0]
        confidence  = round(float(max(proba)) * 100, 1)

        # Determine anomaly type from readings
        voltage = data.get('voltage_v', 220)
        current = data.get('current_a', 0)
        power   = data.get('power_w', 0)
        exp_max = data.get('expected_max_w', 100)

        if prediction == 1:
            if voltage > 235:   anomaly_type = 'voltage_spike'
            elif voltage < 200: anomaly_type = 'voltage_drop'
            elif power > exp_max * 1.3: anomaly_type = 'current_overload'
            elif voltage < 100: anomaly_type = 'short_circuit'
            else:               anomaly_type = 'device_mismatch'
            severity = 'critical' if voltage < 100 else 'high' if voltage > 235 or power > exp_max * 1.5 else 'medium'
        else:
            anomaly_type = 'normal'
            severity     = 'none'

        return jsonify({
            'success':      True,
            'is_anomaly':   bool(prediction),
            'anomaly_type': anomaly_type,
            'severity':     severity,
            'confidence':   confidence,
            'message':      f'⚠️ {anomaly_type.replace("_"," ").title()} detected!' if prediction == 1 else '✅ Normal reading'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── MODEL 4: Device Detection ────────────────────────────────
@app.route('/predict/device', methods=['POST'])
def predict_device():
    try:
        data = request.json
        features = [[
            data.get('voltage_v', 220),
            data.get('current_a', 0),
            data.get('power_w', 0),
            data.get('hour_of_day', 12),
            SLOT_MAP.get(data.get('time_slot', 'normal'), 0),
        ]]
        features_sc  = scaler4.transform(features)
        pred_enc     = model4.predict(features_sc)[0]
        device_name  = le_device.inverse_transform([pred_enc])[0]
        proba        = model4.predict_proba(features_sc)[0]
        confidence   = round(float(max(proba)) * 100, 1)

        # Top 3 possible devices
        top3_idx   = np.argsort(proba)[-3:][::-1]
        top3       = [{'device': le_device.inverse_transform([i])[0],
                       'confidence': round(float(proba[i])*100,1)} for i in top3_idx]

        return jsonify({
            'success':     True,
            'detected_device': device_name,
            'confidence':  confidence,
            'top3':        top3,
            'power_w':     data.get('power_w', 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── MODEL 5: Peak Saving ─────────────────────────────────────
@app.route('/predict/saving', methods=['POST'])
def predict_saving():
    try:
        data = request.json

        device_list = list(le_device.classes_)
        device_name = data.get('device_id', 'bulb_led_9w')
        device_enc  = device_list.index(device_name) if device_name in device_list else 0

        time_slot   = data.get('time_slot', 'normal')
        is_peak     = 1 if time_slot == 'peak' else 0
        rate        = {'off_peak':18,'normal':30,'peak':52}.get(time_slot, 30)
        dur         = data.get('on_duration_min', 60)
        power_kwh   = data.get('power_w', 100) / 1000 * (dur / 60)
        cost        = round(power_kwh * rate, 2)

        features = [[
            device_enc,
            data.get('hour_of_day', 12),
            SLOT_MAP.get(time_slot, 0),
            dur,
            is_peak,
            cost,
        ]]
        prediction = model5.predict(features)[0]
        saving_rs  = round(cost * 0.42, 2) if prediction == 1 else 0

        return jsonify({
            'success':        True,
            'saving_possible': bool(prediction),
            'current_cost_rs': cost,
            'potential_saving_rs': saving_rs,
            'message': f'💡 Shift to off-peak! Save Rs {saving_rs}' if prediction == 1 else '✅ Good time to use this device'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── ALL IN ONE: Full Prediction ──────────────────────────────
@app.route('/predict/all', methods=['POST'])
def predict_all():
    try:
        data = request.json
        voltage   = data.get('voltage_v', 220)
        current   = data.get('current_a', 0)
        power     = data.get('power_w', 0)
        time_slot = data.get('time_slot', 'normal')
        hour      = data.get('hour_of_day', 12)

        # Anomaly
        anom_feat = [[voltage, current, power,
                      data.get('expected_min_w',0),
                      data.get('expected_max_w',100),
                      hour, SLOT_MAP.get(time_slot,0)]]
        anom_sc   = scaler3.transform(anom_feat)
        is_anomaly= bool(model3.predict(anom_sc)[0])

        # Device
        dev_feat  = [[voltage, current, power, hour, SLOT_MAP.get(time_slot,0)]]
        dev_sc    = scaler4.transform(dev_feat)
        dev_enc   = model4.predict(dev_sc)[0]
        device    = le_device.inverse_transform([dev_enc])[0]
        dev_conf  = round(float(max(model4.predict_proba(dev_sc)[0]))*100,1)

        # Saving
        is_peak   = 1 if time_slot == 'peak' else 0
        rate      = {'off_peak':18,'normal':30,'peak':52}.get(time_slot,30)
        dur       = data.get('on_duration_min', 60)
        cost      = round((power/1000)*(dur/60)*rate, 2)
        save_feat = [[dev_enc, hour, SLOT_MAP.get(time_slot,0), dur, is_peak, cost]]
        saving    = bool(model5.predict(save_feat)[0])

        return jsonify({
            'success':    True,
            'sensor': {
                'voltage_v': voltage,
                'current_a': current,
                'power_w':   power,
                'time_slot': time_slot
            },
            'anomaly': {
                'is_anomaly': is_anomaly,
                'message': '⚠️ Anomaly detected!' if is_anomaly else '✅ Normal'
            },
            'device': {
                'detected': device,
                'confidence': dev_conf
            },
            'saving': {
                'saving_possible': saving,
                'current_cost_rs': cost,
                'message': f'💡 Save Rs {round(cost*0.42,2)} by shifting to off-peak!' if saving else '✅ Good time'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
