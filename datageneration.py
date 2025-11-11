import pandas as pd
import random
from datetime import datetime, timedelta

# Configuration
start_date = datetime(2025, 11, 1)
end_date = datetime(2025, 11, 30)
machines = [f"M{i}" for i in range(1, 7)] + [f"D{i}" for i in range(1, 7)]

records = []

current_date = start_date
while current_date <= end_date:
    day_name = current_date.strftime("%a")
    for m in machines:
        machine_type = "Washer" if m.startswith("M") else "Dryer"
        num_uses = random.randint(3, 6)
        time_pointer = datetime.combine(current_date, datetime.strptime("07:00", "%H:%M").time())

        for _ in range(num_uses):
            # Random gap before next use
            gap = random.randint(20, 90)
            time_pointer += timedelta(minutes=gap)
            if time_pointer.hour >= 23:
                break

            # Duration rules
            if machine_type == "Washer":
                duration = 30
            else:
                duration = 30 + random.choice([0, 4, 8, 12, 16, 20, 24, 28, 30])

            start_time_str = time_pointer.strftime("%H:%M")
            records.append([m, machine_type, current_date.strftime("%Y-%m-%d"), day_name,
                            start_time_str, duration, "Running"])

            # Move pointer to end of cycle
            time_pointer += timedelta(minutes=duration)

            # Add a short idle entry
            idle_gap = random.randint(10, 40)
            idle_start = time_pointer
            records.append([m, machine_type, current_date.strftime("%Y-%m-%d"), day_name,
                            idle_start.strftime("%H:%M"), idle_gap, "Idle"])
            time_pointer += timedelta(minutes=idle_gap)
    current_date += timedelta(days=1)

# Save to CSV
df = pd.DataFrame(records, columns=[
    "Machine_ID", "MachineType", "Date", "Day", "Start_Time", "Duration(min)", "Status"
])
df.to_csv("laundry_usage.csv", index=False)
print("✅ Dataset generated: laundry_usage.csv")
print(df.head())
