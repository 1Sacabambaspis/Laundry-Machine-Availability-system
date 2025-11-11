import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QMessageBox, QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QFont, QLinearGradient, QBrush, QColor, QPainter
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import joblib
from PyQt5.QtWidgets import QTimeEdit
from PyQt5.QtCore import QTime

# Load model, encoders, and scaler
model = joblib.load("laundry_status_model.pkl")
le_day = joblib.load("le_day.pkl")
le_machine = joblib.load("le_machine.pkl")
le_type = joblib.load("le_type.pkl")
scaler = joblib.load("scaler.pkl")


class LaundryPredictorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laundry Machine Availability Predictor")
        self.resize(550, 500)
        self.setFont(QFont("Helvetica Neue", 11))
        self.init_ui()

    def paintEvent(self, event):
        """Gradient background"""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#8ecae6"))
        gradient.setColorAt(0.5, QColor("#219ebc"))
        gradient.setColorAt(1.0, QColor("#023047"))
        painter.fillRect(self.rect(), QBrush(gradient))

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)

        # Card
        card = QWidget()
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(12)

        # Shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        card.setGraphicsEffect(shadow)

        card.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.75);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QLabel {
                color: #ffb703;
                font-weight: 500;
            }
            QComboBox, QSpinBox {
                color: #ffb703;
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid #023047;
                border-radius: 6px;
                padding: 4px 6px;
            }
        """)

        # Title
        title = QLabel("Laundry Availability Prediction System")
        title.setFont(QFont("Georgia", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #219ebc; margin-bottom: 10px;")
        card_layout.addWidget(line)

        # Washer Machine
        self.washer_machine = QComboBox()
        self.washer_machine.addItems(["None","M1","M2","M3","M4","M5","M6"])
        card_layout.addWidget(QLabel("Select Washer (optional):"))
        card_layout.addWidget(self.washer_machine)

        self.washer_duration = QSpinBox()
        self.washer_duration.setRange(0, 120)
        self.washer_duration.setSingleStep(10)
        card_layout.addWidget(QLabel("Washer Duration (minutes):"))
        card_layout.addWidget(self.washer_duration)

        # Dryer Machine
        self.dryer_machine = QComboBox()
        self.dryer_machine.addItems(["None","D1","D2","D3","D4","D5","D6"])
        card_layout.addWidget(QLabel("Select Dryer (optional):"))
        card_layout.addWidget(self.dryer_machine)

        self.dryer_duration = QSpinBox()
        self.dryer_duration.setRange(0, 120)
        self.dryer_duration.setSingleStep(10)
        card_layout.addWidget(QLabel("Dryer Duration (minutes):"))
        card_layout.addWidget(self.dryer_duration)

        # Day
        self.day_dropdown = QComboBox()
        self.day_dropdown.addItems(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        card_layout.addWidget(QLabel("Select Day of Week:"))
        card_layout.addWidget(self.day_dropdown)

        # Hour
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("hh:mm AP")  # shows 12-hour format with AM/PM
        self.time_edit.setTime(QTime.currentTime())  # default to current time
        card_layout.addWidget(QLabel("Select Time:"))
        card_layout.addWidget(self.time_edit)
        


        # Predict Button
        self.predict_button = QPushButton("Predict Availability")
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #fb8500;
                color: white;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffb703;
                color: #023047;
            }
        """)
        self.predict_button.clicked.connect(self.make_prediction)
        card_layout.addWidget(self.predict_button)

        # Result Label
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Georgia", 13, QFont.Bold))
        self.result_label.setStyleSheet("color: #ffb703;")
        card_layout.addWidget(self.result_label)

        card.setLayout(card_layout)
        main_layout.addWidget(card)

    def make_prediction(self):
        try:
            # Get selected day and time
            day = self.day_dropdown.currentText()
            time = self.time_edit.time()
            decimal_hour = time.hour() + time.minute() / 60.0  # convert to decimal hours

            # Determine if high-demand day (Wed, Sat, Sun)
            high_demand = 1 if day in ["Wed", "Sat", "Sun"] else 0

            result_text = ""

            # -----------------------------
            # Washer prediction
            # -----------------------------
            washer_use = self.washer_machine.currentText() != "None"
            washer_duration = self.washer_duration.value() if washer_use else 0

            if washer_use:
                features_df = pd.DataFrame(
                    [[
                        le_machine.transform([self.washer_machine.currentText()])[0],
                        le_type.transform(["Washer"])[0],
                        le_day.transform([day])[0],
                        decimal_hour,
                        washer_duration
                    ]],
                    columns=["Machine_ID_encoded", "MachineType_encoded", "DayOfWeek_encoded", "Hour", "Duration(min)"]
                )

                # Scale numeric features
                features_df[["Hour", "Duration(min)"]] = scaler.transform(features_df[["Hour", "Duration(min)"]])

                # Predict
                pred = model.predict(features_df)[0]
                status = "RUNNING" if pred == 1 else "AVAILABLE"
                result_text += f"Washer: {status}\n"

            # -----------------------------
            # Dryer prediction
            # -----------------------------
            dryer_use = self.dryer_machine.currentText() != "None"
            dryer_duration = self.dryer_duration.value() if dryer_use else 0

            if dryer_use:
                features_df = pd.DataFrame(
                    [[
                        le_machine.transform([self.dryer_machine.currentText()])[0],
                        le_type.transform(["Dryer"])[0],
                        le_day.transform([day])[0],
                        decimal_hour,
                        dryer_duration
                    ]],
                    columns=["Machine_ID_encoded", "MachineType_encoded", "DayOfWeek_encoded", "Hour", "Duration(min)"]
                )

                # Scale numeric features
                features_df[["Hour", "Duration(min)"]] = scaler.transform(features_df[["Hour", "Duration(min)"]])

                # Predict
                pred = model.predict(features_df)[0]
                status = "RUNNING" if pred == 1 else "AVAILABLE"
                result_text += f"Dryer: {status}"

            if not washer_use and not dryer_use:
                result_text = "Please select at least one machine to predict."

            # Display result
            self.result_label.setText(result_text)

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LaundryPredictorUI()
    window.show()
    sys.exit(app.exec_())
