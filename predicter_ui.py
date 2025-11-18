import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QMessageBox, QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
    QGridLayout, QTimeEdit
)
from PyQt5.QtGui import QFont, QLinearGradient, QBrush, QColor, QPainter
from PyQt5.QtCore import Qt, QTime, QTimer

# Load enhanced AI model and artifacts
try:
    model = joblib.load("laundry_status_enhanced_model.pkl")
    artifacts = joblib.load("laundry_model_artifacts.pkl")
    
    scaler = artifacts['scaler']
    le_day = artifacts['label_encoders']['day']
    le_machine = artifacts['label_encoders']['machine']
    le_type = artifacts['label_encoders']['type']
    le_time_period = artifacts['label_encoders']['time_period']
    le_duration_cat = artifacts['label_encoders']['duration_category']
    feature_names = artifacts['feature_names']
    
    print("Enhanced model loaded successfully!")
    print(f"Model trained on: {artifacts['model_metadata']['training_date']}")
    print(f"Features: {len(feature_names)}")
except Exception as e:
    print(f"Error loading enhanced model: {e}")
    print("Falling back to basic model...")
    model = joblib.load("laundry_status_model.pkl")
    le_day = joblib.load("le_day.pkl")
    le_machine = joblib.load("le_machine.pkl")
    le_type = joblib.load("le_type.pkl")
    scaler = joblib.load("scaler.pkl")
    artifacts = None
    feature_names = None


class GlowingStatusBox(QLabel):
    """Status box with native PyQt5 glow effects"""
    def __init__(self, machine_name, parent=None):
        super().__init__(parent)
        self.machine_name = machine_name
        self.status = "unknown"
        self.probability = 0.0
        self._glow_radius = 20
        self.setText(machine_name)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(110, 85)
        self.setFont(QFont("Arial", 9, QFont.Bold))
        
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 3)
        self.setGraphicsEffect(self.shadow)
        
        self.glow_timer = QTimer(self)
        self.glow_timer.timeout.connect(self.update_glow)
        self.glow_direction = 1
        
        self.update_style()
    
    def set_status(self, status, probability=0.0):
        self.status = status
        self.probability = probability
        
        status_display = {
            "available": "✓ OPEN",
            "running": "⊗ BUSY", 
            "recommended": "★ BEST",
            "unknown": "—"
        }
        
        prob_text = f"\n{int(probability*100)}%" if probability > 0 else ""
        self.setText(f"{self.machine_name}\n{status_display.get(status, '—')}{prob_text}")
        
        if status == "recommended":
            self.glow_timer.start(30)
        else:
            self.glow_timer.stop()
            self._glow_radius = 20
        
        self.update_style()
    
    def update_glow(self):
        self._glow_radius += 3 * self.glow_direction
        if self._glow_radius >= 45:
            self.glow_direction = -1
        elif self._glow_radius <= 20:
            self.glow_direction = 1
        self.shadow.setBlurRadius(self._glow_radius)
    
    def update_style(self):
        styles = {
            "available": {
                "bg": "#c6f6d5",
                "border": "#10b981",
                "text": "#065f46",
                "shadow": QColor(16, 185, 129, 140)
            },
            "running": {
                "bg": "#fecaca",
                "border": "#ef4444",
                "text": "#991b1b",
                "shadow": QColor(239, 68, 68, 140)
            },
            "recommended": {
                "bg": "#fde68a",
                "border": "#f59e0b",
                "text": "#92400e",
                "shadow": QColor(245, 158, 11, 200)
            },
            "unknown": {
                "bg": "#e2e8f0",
                "border": "#94a3b8",
                "text": "#475569",
                "shadow": QColor(148, 163, 184, 100)
            }
        }
        
        style = styles.get(self.status, styles["unknown"])
        self.shadow.setColor(style['shadow'])
        
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {style['bg']};
                border: 3px solid {style['border']};
                border-radius: 12px;
                color: {style['text']};
                padding: 8px;
                font-weight: bold;
            }}
        """)


class ModernButton(QPushButton):
    def __init__(self, text, color1, color2, parent=None):
        super().__init__(text, parent)
        self.color1 = QColor(color1)
        self.color2 = QColor(color2)
        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
        
        self.apply_style()
    
    def apply_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.color1.name()}, stop:1 {self.color2.name()});
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.color2.name()}, stop:1 {self.color1.name()});
            }}
            QPushButton:pressed {{
                padding-top: 14px;
                padding-bottom: 10px;
            }}
        """)


class StyledComboBox(QComboBox):
    def __init__(self, border_color, parent=None):
        super().__init__(parent)
        self.border_color = border_color
        self.setFixedHeight(35)
        self.apply_style()
    
    def apply_style(self):
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: white;
                border: 2px solid {self.border_color};
                border-radius: 8px;
                padding: 6px 10px;
                color: #1e293b;
                font-weight: 500;
            }}
            QComboBox:hover {{
                border: 2px solid {QColor(self.border_color).darker(110).name()};
                background-color: #f8fafc;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 25px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {self.border_color};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: white;
                border: 2px solid {self.border_color};
                selection-background-color: {QColor(self.border_color).lighter(180).name()};
                selection-color: #1e293b;
            }}
        """)


class StyledSpinBox(QSpinBox):
    def __init__(self, border_color, parent=None):
        super().__init__(parent)
        self.border_color = border_color
        self.setFixedHeight(35)
        self.apply_style()
    
    def apply_style(self):
        self.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 2px solid {self.border_color};
                border-radius: 8px;
                padding: 6px 10px;
                color: #1e293b;
                font-weight: 500;
            }}
            QSpinBox:hover {{
                border: 2px solid {QColor(self.border_color).darker(110).name()};
                background-color: #f8fafc;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background: transparent;
                border: none;
                width: 18px;
            }}
        """)


class LaundryPredictorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laundry Predicting System")
        self.setMinimumSize(850, 1000)
        self.resize(900, 750)
        self.status_boxes = {}
        self.init_ui()

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#1e293b"))
        gradient.setColorAt(0.5, QColor("#334155"))
        gradient.setColorAt(1.0, QColor("#0f172a"))
        painter.fillRect(self.rect(), QBrush(gradient))

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        card = QWidget()
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(25, 25, 25, 25)
        card_layout.setSpacing(15)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 150))
        card.setGraphicsEffect(shadow)

        card.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.98),
                    stop:0.3 rgba(248, 250, 252, 0.95),
                    stop:0.7 rgba(241, 245, 249, 0.95),
                    stop:1 rgba(226, 232, 240, 0.92));
                border-radius: 18px;
            }
            QLabel {
                color: #0f172a;
                background: transparent;
            }
        """)

        title = QLabel("Laundry Predicting System")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0f172a; margin-bottom: 5px;")
        card_layout.addWidget(title)

        subtitle = QLabel("Smart Availability Predictions & Time Recommendations")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #64748b; margin-bottom: 12px;")
        card_layout.addWidget(subtitle)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #e2e8f0; max-height: 2px;")
        card_layout.addWidget(divider)

        status_label = QLabel("Machine Status")
        status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_label.setStyleSheet("color: #1e293b; margin-top: 8px;")
        card_layout.addWidget(status_label)

        status_grid = QGridLayout()
        status_grid.setSpacing(10)
        status_grid.setContentsMargins(0, 10, 0, 10)
        
        w_header = QLabel("WASHERS")
        w_header.setFont(QFont("Arial", 9, QFont.Bold))
        w_header.setAlignment(Qt.AlignCenter)
        w_header.setStyleSheet("color: #3b82f6; margin-bottom: 5px;")
        status_grid.addWidget(w_header, 0, 0, 1, 6)
        
        for i, machine in enumerate(["M1", "M2", "M3", "M4", "M5", "M6"]):
            box = GlowingStatusBox(machine)
            self.status_boxes[machine] = box
            status_grid.addWidget(box, 1, i)
        
        d_header = QLabel("DRYERS")
        d_header.setFont(QFont("Arial", 9, QFont.Bold))
        d_header.setAlignment(Qt.AlignCenter)
        d_header.setStyleSheet("color: #8b5cf6; margin-top: 8px; margin-bottom: 5px;")
        status_grid.addWidget(d_header, 2, 0, 1, 6)
        
        for i, machine in enumerate(["D1", "D2", "D3", "D4", "D5", "D6"]):
            box = GlowingStatusBox(machine)
            self.status_boxes[machine] = box
            status_grid.addWidget(box, 3, i)
        
        card_layout.addLayout(status_grid)

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.HLine)
        divider2.setStyleSheet("background-color: #e2e8f0; max-height: 1px;")
        card_layout.addWidget(divider2)

        select_label = QLabel("Configure Session")
        select_label.setFont(QFont("Arial", 12, QFont.Bold))
        select_label.setStyleSheet("color: #1e293b;")
        card_layout.addWidget(select_label)

        machine_layout = QHBoxLayout()
        machine_layout.setSpacing(20)

        washer_col = QVBoxLayout()
        washer_col.setSpacing(8)
        
        w_label = QLabel("Washer")
        w_label.setFont(QFont("Arial", 10, QFont.Bold))
        w_label.setStyleSheet("color: #3b82f6;")
        washer_col.addWidget(w_label)
        
        self.washer_machine = StyledComboBox("#3b82f6")
        self.washer_machine.addItems(["None","M1","M2","M3","M4","M5","M6"])
        self.washer_machine.setToolTip("Select washer machine")
        washer_col.addWidget(self.washer_machine)
        
        dur_label1 = QLabel("Duration (min)")
        dur_label1.setStyleSheet("color: #64748b; font-size: 8pt;")
        washer_col.addWidget(dur_label1)
        
        self.washer_duration = StyledSpinBox("#3b82f6")
        self.washer_duration.setRange(0, 120)
        self.washer_duration.setValue(30)
        self.washer_duration.setSingleStep(5)
        self.washer_duration.setToolTip("Wash time (0-120 min)")
        washer_col.addWidget(self.washer_duration)
        machine_layout.addLayout(washer_col)

        dryer_col = QVBoxLayout()
        dryer_col.setSpacing(8)
        
        d_label = QLabel("Dryer")
        d_label.setFont(QFont("Arial", 10, QFont.Bold))
        d_label.setStyleSheet("color: #8b5cf6;")
        dryer_col.addWidget(d_label)
        
        self.dryer_machine = StyledComboBox("#8b5cf6")
        self.dryer_machine.addItems(["None","D1","D2","D3","D4","D5","D6"])
        self.dryer_machine.setToolTip("Select dryer machine")
        dryer_col.addWidget(self.dryer_machine)
        
        dur_label2 = QLabel("Duration (min)")
        dur_label2.setStyleSheet("color: #64748b; font-size: 8pt;")
        dryer_col.addWidget(dur_label2)
        
        self.dryer_duration = StyledSpinBox("#8b5cf6")
        self.dryer_duration.setRange(0, 120)
        self.dryer_duration.setValue(45)
        self.dryer_duration.setSingleStep(5)
        self.dryer_duration.setToolTip("Dry time (0-120 min)")
        dryer_col.addWidget(self.dryer_duration)
        machine_layout.addLayout(dryer_col)
        
        card_layout.addLayout(machine_layout)

        datetime_layout = QHBoxLayout()
        datetime_layout.setSpacing(20)
        
        day_col = QVBoxLayout()
        day_col.setSpacing(8)
        day_label = QLabel("Day")
        day_label.setFont(QFont("Arial", 10, QFont.Bold))
        day_label.setStyleSheet("color: #64748b;")
        day_col.addWidget(day_label)
        
        self.day_dropdown = StyledComboBox("#10b981")
        self.day_dropdown.addItems(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        self.day_dropdown.setCurrentIndex(datetime.now().weekday())
        self.day_dropdown.setToolTip("Day of the week")
        day_col.addWidget(self.day_dropdown)
        datetime_layout.addLayout(day_col)
        
        time_col = QVBoxLayout()
        time_col.setSpacing(8)
        time_label = QLabel("Time")
        time_label.setFont(QFont("Arial", 10, QFont.Bold))
        time_label.setStyleSheet("color: #64748b;")
        time_col.addWidget(time_label)
        
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("hh:mm AP")
        self.time_edit.setTime(QTime.currentTime())
        self.time_edit.setFixedHeight(35)
        self.time_edit.setStyleSheet("""
            QTimeEdit {
                background-color: white;
                border: 2px solid #10b981;
                border-radius: 8px;
                padding: 6px 10px;
                color: #1e293b;
                font-weight: 500;
            }
            QTimeEdit:hover {
                border: 2px solid #059669;
                background-color: #f8fafc;
            }
        """)
        self.time_edit.setToolTip("Start time")
        time_col.addWidget(self.time_edit)
        datetime_layout.addLayout(time_col)
        
        card_layout.addLayout(datetime_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.predict_btn = ModernButton("Check", "#3b82f6", "#2563eb")
        self.predict_btn.clicked.connect(self.make_prediction)
        self.predict_btn.setToolTip("Check availability now")
        
        self.recommend_btn = ModernButton("Find Best", "#8b5cf6", "#7c3aed")
        self.recommend_btn.clicked.connect(self.recommend_time)
        self.recommend_btn.setToolTip("Find optimal time slot")
        
        self.check_all_btn = ModernButton("View All", "#10b981", "#059669")
        self.check_all_btn.clicked.connect(self.check_all_machines)
        self.check_all_btn.setToolTip("Check all machines")
        
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.recommend_btn)
        button_layout.addWidget(self.check_all_btn)
        
        card_layout.addLayout(button_layout)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Courier", 10))
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(100)
        self.result_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc,
                    stop:1 #e2e8f0);
                border: 2px solid #cbd5e1;
                border-radius: 12px;
                padding: 15px;
                color: #1e293b;
            }
        """)
        card_layout.addWidget(self.result_label)

        card.setLayout(card_layout)
        main_layout.addWidget(card)

    def create_enhanced_features(self, machine_id, machine_type, day, hour, duration):
        """Create enhanced feature set with all engineered features"""
        try:
            if not artifacts:
                # Fallback to basic features
                machine_id_enc = le_machine.transform([machine_id])[0]
                machine_type_enc = le_type.transform([machine_type])[0]
                day_enc = le_day.transform([day])[0]
                
                features_df = pd.DataFrame([[
                    machine_id_enc, machine_type_enc, day_enc, hour, duration
                ]], columns=["Machine_ID_encoded", "MachineType_encoded",
                            "DayOfWeek_encoded", "Hour", "Duration(min)"])
                return features_df
            
            # Enhanced features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            is_weekend = 1 if day in ["Sat", "Sun"] else 0
            is_weekday = 1 - is_weekend
            high_demand_day = 1 if day in ["Wed", "Sat", "Sun"] else 0
            
            if 6 <= hour < 12:
                time_period = "Morning"
            elif 12 <= hour < 18:
                time_period = "Afternoon"
            elif 18 <= hour < 22:
                time_period = "Evening"
            else:
                time_period = "Night"
            
            is_peak_hour = 1 if (9 <= hour <= 11) or (18 <= hour <= 21) else 0
            is_long_cycle = 1 if duration > 40 else 0
            
            if duration < 30:
                duration_cat = "Short"
            elif duration < 50:
                duration_cat = "Medium"
            else:
                duration_cat = "Long"
            
            peak_weekend = is_peak_hour * is_weekend
            demand_peak = high_demand_day * is_peak_hour
            
            machine_id_enc = le_machine.transform([machine_id])[0]
            machine_type_enc = le_type.transform([machine_type])[0]
            day_enc = le_day.transform([day])[0]
            time_period_enc = le_time_period.transform([time_period])[0]
            duration_cat_enc = le_duration_cat.transform([duration_cat])[0]
            
            features = {
                "Machine_ID_encoded": machine_id_enc,
                "MachineType_encoded": machine_type_enc,
                "DayOfWeek_encoded": day_enc,
                "Hour": hour,
                "Hour_Sin": hour_sin,
                "Hour_Cos": hour_cos,
                "Duration(min)": duration,
                "High_Demand_Day": high_demand_day,
                "Is_Weekend": is_weekend,
                "Is_Weekday": is_weekday,
                "Is_Peak_Hour": is_peak_hour,
                "Time_Period_encoded": time_period_enc,
                "Is_Long_Cycle": is_long_cycle,
                "Duration_Cat_encoded": duration_cat_enc,
                "Peak_Weekend": peak_weekend,
                "Demand_Peak": demand_peak
            }
            
            features_df = pd.DataFrame([features], columns=feature_names)
            return features_df
            
        except Exception as e:
            print(f"Feature creation error: {e}")
            return None
    
    def predict_machine(self, machine_id, machine_type, day, decimal_hour, duration):
        """Make prediction using enhanced model with multi-output support"""
        try:
            features_df = self.create_enhanced_features(
                machine_id, machine_type, day, decimal_hour, duration
            )
            
            if features_df is None:
                return None, 0.0
            
            features_scaled = scaler.transform(features_df)
            features_df_scaled = pd.DataFrame(
                features_scaled, 
                columns=features_df.columns
            )
            
            prediction = model.predict(features_df_scaled)[0]
            
            if artifacts:
                # Multi-output model
                if machine_type == "Washer":
                    pred_status = prediction[0]
                    estimator_idx = 0
                else:
                    pred_status = prediction[1]
                    estimator_idx = 1
                
                try:
                    proba = model.estimators_[estimator_idx].predict_proba(features_df_scaled)[0]
                    running_prob = proba[1] if len(proba) > 1 else proba[0]
                except:
                    running_prob = 1.0 if pred_status == 1 else 0.0
            else:
                # Basic model
                pred_status = prediction
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_df_scaled)[0]
                        running_prob = proba[1] if len(proba) > 1 else proba[0]
                    else:
                        running_prob = 1.0 if pred_status == 1 else 0.0
                except:
                    running_prob = 1.0 if pred_status == 1 else 0.0
            
            return int(pred_status), float(running_prob)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    def make_prediction(self):
        """Enhanced prediction with better probability handling"""
        try:
            day = self.day_dropdown.currentText()
            time = self.time_edit.time()
            decimal_hour = time.hour() + time.minute() / 60.0
            
            results = []
            results.append(f"Location: {day} at {time.toString('hh:mm AP')}\n")

            if self.washer_machine.currentText() != "None":
                machine = self.washer_machine.currentText()
                duration = self.washer_duration.value()
                pred, prob = self.predict_machine(machine, "Washer", day, decimal_hour, duration)
                
                if pred is not None:
                    icon = "BUSY" if pred == 1 else "OPEN"
                    confidence = int(prob * 100) if pred == 1 else int((1-prob) * 100)
                    results.append(f"Washer {machine}: {icon} ({confidence}% confidence)")
                    
                    if pred == 0:
                        results.append(f"   Good timing! Duration: {duration} min\n")
                    else:
                        results.append(f"   High occupancy expected\n")

            if self.dryer_machine.currentText() != "None":
                machine = self.dryer_machine.currentText()
                duration = self.dryer_duration.value()
                pred, prob = self.predict_machine(machine, "Dryer", day, decimal_hour, duration)
                
                if pred is not None:
                    icon = "BUSY" if pred == 1 else "OPEN"
                    confidence = int(prob * 100) if pred == 1 else int((1-prob) * 100)
                    results.append(f"Dryer {machine}: {icon} ({confidence}% confidence)")
                    
                    if pred == 0:
                        results.append(f"   Good timing! Duration: {duration} min")
                    else:
                        results.append(f"   High occupancy expected")

            if len(results) == 1:
                results.append("Select at least one machine")

            self.result_label.setText("\n".join(results))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Prediction failed: {e}")

    def recommend_time(self):
        """Enhanced recommendation with better slot finding"""
        try:
            day = self.day_dropdown.currentText()
            current_time = self.time_edit.time()
            start_hour = current_time.hour() + current_time.minute() / 60.0
            
            results = []
            results.append(f"Finding best times for {day}...\n")

            if self.washer_machine.currentText() != "None":
                machine = self.washer_machine.currentText()
                duration = self.washer_duration.value()
                slot = self.find_next_slot(machine, "Washer", day, start_hour, duration)
                
                if slot:
                    hr, mn = slot
                    results.append(f"Washer {machine}:")
                    results.append(f"   Best time: {hr:02d}:{mn:02d} ({duration} min)\n")
                    self.status_boxes[machine].set_status("recommended", 0.92)
                else:
                    results.append(f"Washer {machine}: No optimal slots found\n")

            if self.dryer_machine.currentText() != "None":
                machine = self.dryer_machine.currentText()
                duration = self.dryer_duration.value()
                
                if self.washer_machine.currentText() != "None":
                    dryer_start = start_hour + (self.washer_duration.value() / 60.0)
                else:
                    dryer_start = start_hour
                
                slot = self.find_next_slot(machine, "Dryer", day, dryer_start, duration)
                
                if slot:
                    hr, mn = slot
                    results.append(f"Dryer {machine}:")
                    results.append(f"   Best time: {hr:02d}:{mn:02d} ({duration} min)")
                    self.status_boxes[machine].set_status("recommended", 0.92)
                else:
                    results.append(f"Dryer {machine}: No optimal slots found")

            if len(results) == 1:
                results.append("Select at least one machine")
            
            high_demand = ["Wed", "Sat", "Sun"]
            if day in high_demand:
                results.append("\nTip: Try early morning (6-8 AM) or late evening (9-11 PM) on busy days")

            self.result_label.setText("\n".join(results))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Recommendation failed: {e}")

    def find_next_slot(self, machine, machine_type, day, start_hour, duration):
        """Find next available time slot with lower occupancy threshold"""
        for h in np.arange(start_hour, 24, 0.25):
            pred, prob = self.predict_machine(machine, machine_type, day, h, duration)
            
            if pred is not None and pred == 0 and prob < 0.4:
                return (int(h), int((h - int(h)) * 60))
        
        return None

    def check_all_machines(self):
        """Check all machines with enhanced predictions"""
        try:
            day = self.day_dropdown.currentText()
            time = self.time_edit.time()
            decimal_hour = time.hour() + time.minute() / 60.0
            
            for box in self.status_boxes.values():
                box.set_status("unknown")
            
            for machine in ["M1", "M2", "M3", "M4", "M5", "M6"]:
                pred, prob = self.predict_machine(machine, "Washer", day, decimal_hour, 30)
                if pred is not None:
                    status = "running" if pred == 1 else "available"
                    self.status_boxes[machine].set_status(status, prob)
            
            for machine in ["D1", "D2", "D3", "D4", "D5", "D6"]:
                pred, prob = self.predict_machine(machine, "Dryer", day, decimal_hour, 45)
                if pred is not None:
                    status = "running" if pred == 1 else "available"
                    self.status_boxes[machine].set_status(status, prob)
            
            self.result_label.setText(
                f"All machines checked for {day} at {time.toString('hh:mm AP')}\n\n"
                f"Green = Available | Red = Busy | Gold = Recommended\n"
                f"Percentages show prediction confidence"
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Check failed: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LaundryPredictorUI()
    window.show()
    sys.exit(app.exec_())