import os
import uuid
import base64
import io
import hashlib
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from functools import wraps

from ml_utils import predictor

app = Flask(__name__)
app.secret_key = "agrovision_enterprise_key_9988"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agrovision_web.db'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # 64MB for batch uploads

db = SQLAlchemy(app)

# --- Jinja Filters ---
@app.template_filter('from_json')
def from_json_filter(s):
    return json.loads(s)

# --- Enhanced Database Models ---

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), index=True) # For Grouping Batch Scans
    plant_part = db.Column(db.String(50))
    disease = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(20))
    image_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Advanced Data
    metrics_json = db.Column(db.Text)
    prescription_json = db.Column(db.Text) # Smart Auto-Prescription

with app.app_context():
    db.create_all()
    # Migration Safety Unit
    inspector = db.inspect(db.engine)
    columns = [c['name'] for c in inspector.get_columns('prediction')]
    
    migrations = {
        'batch_id': "ALTER TABLE prediction ADD COLUMN batch_id VARCHAR(50)",
        'prescription_json': "ALTER TABLE prediction ADD COLUMN prescription_json TEXT",
        'metrics_json': "ALTER TABLE prediction ADD COLUMN metrics_json TEXT"
    }
    
    for col, sql in migrations.items():
        if col not in columns:
            with db.engine.connect() as conn:
                conn.execute(db.text(sql))
                conn.commit()
                print(f"Migration: Added {col} to database.")

# --- Smart Auto-Prescription Engine ---

def generate_prescription(disease, severity):
    # Professional Dosage & Frequency Logic
    prescriptions = {
        "Blight": {
            "regimen": "Systemic Fungicide Protocol",
            "chemical": "Chlorothalonil (500g/L)",
            "dosage": "20ml per 10L water",
            "frequency": "Every 7 days for 3 weeks",
            "organic": "Liquid Copper Spray (0.5%)",
            "notes": "Apply in early morning. Remove heavily infected lower leaves."
        },
        "Spot": {
            "regimen": "Protective Coating Protocol",
            "chemical": "Mancozeb (75% WP)",
            "dosage": "25g per 15L water",
            "frequency": "Every 10 days until clear",
            "organic": "Baking Soda & Neem Solution",
            "notes": "Ensure coverage on leaf undersides."
        },
        "Rot": {
            "regimen": "Soil Sterilization Protocol",
            "chemical": "Fosetyl-Aluminium",
            "dosage": "15g per 5L water (Soil Drench)",
            "frequency": "Single application, repeat in 14 days",
            "organic": "Trichoderma viride bio-fungicide",
            "notes": "Reduce irrigation by 40%. Improve drainage."
        },
        "Healthy": {
            "regimen": "Maintenance & Growth Protocol",
            "chemical": "N-P-K 20-20-20 (Micro-dosage)",
            "dosage": "10g per 10L water",
            "frequency": "Monthly",
            "organic": "Seaweed Extract / Fish Emulsion",
            "notes": "Monitor for early signs of stress."
        }
    }
    
    # Fallback to Default
    base = prescriptions.get("Healthy") if "healthy" in disease.lower() else prescriptions.get("Blight")
    for key in prescriptions:
        if key.lower() in disease.lower():
            base = prescriptions[key]
            break
            
    # Adjust dosage based on severity
    if severity == "High":
        base["notes"] = "CRITICAL: " + base["notes"] + " Increase application frequency to every 5 days."
        
    return base

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'password':
            session['user'] = 'admin'
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('file')
    if not files or files[0].filename == '': return redirect(url_for('index'))
    
    batch_id = str(uuid.uuid4())[:8]
    predictions = []
    
    for file in files:
        img_bytes = file.read()
        filename = secure_filename(f"{uuid.uuid4()}.jpg")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save image
        with open(file_path, 'wb') as f: f.write(img_bytes)
        
        # Analyze
        result = predictor.predict(img_bytes)
        prescription = generate_prescription(result['disease'], result['severity'])
        
        new_pred = Prediction(
            batch_id=batch_id,
            plant_part=result['plant_part'],
            disease=result['disease'],
            confidence=result['confidence'],
            severity=result['severity'],
            image_path=filename,
            metrics_json=json.dumps(result.get('analytics', {})),
            prescription_json=json.dumps(prescription)
        )
        db.session.add(new_pred)
        predictions.append(new_pred)
    
    db.session.commit()
    
    # If single file, show standard result, if multiple, show Batch Report
    if len(predictions) == 1:
        return render_template('result.html', result=predictions[0], metrics=result.get('analytics', {}), prescription=prescription)
    else:
        return redirect(url_for('batch_report', batch_id=batch_id))

@app.route('/batch-report/<batch_id>')
def batch_report(batch_id):
    results = Prediction.query.filter_by(batch_id=batch_id).all()
    if not results: return redirect(url_for('index'))
    
    # Calculate Batch Stats
    total = len(results)
    healthy = len([r for r in results if 'healthy' in r.disease.lower()])
    infected = total - healthy
    high_risk = len([r for r in results if r.severity == 'High'])
    
    summary = {
        "total": total,
        "healthy": healthy,
        "infected": infected,
        "high_risk": high_risk,
        "health_rate": (healthy / total * 100) if total > 0 else 0
    }
    
    return render_template('batch_report.html', results=results, summary=summary, batch_id=batch_id)

@app.route('/history')
def history():
    if 'user' not in session: return redirect(url_for('login'))
    # Group by batch for history
    from sqlalchemy import func
    batch_data = db.session.query(
        Prediction.batch_id, 
        func.count(Prediction.id), 
        func.max(Prediction.timestamp)
    ).group_by(Prediction.batch_id).order_by(func.max(Prediction.timestamp).desc()).all()
    return render_template('history.html', batches=batch_data)

@app.route('/delete-batch/<batch_id>')
def delete_batch(batch_id):
    preds = Prediction.query.filter_by(batch_id=batch_id).all()
    for p in preds:
        try: os.remove(os.path.join(app.config['UPLOAD_FOLDER'], p.image_path))
        except: pass
        db.session.delete(p)
    db.session.commit()
    return redirect(url_for('history'))

@app.route('/dashboard-stats')
def dashboard_stats():
    total = Prediction.query.count()
    return jsonify({"total": total, "disease_labels": [], "disease_data": [], "severity_high": 0, "severity_medium": 0, "severity_low": 0})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
