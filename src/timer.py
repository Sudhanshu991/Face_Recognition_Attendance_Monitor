from datetime import datetime, timedelta

class PresenceTracker:
    def __init__(self, required_minutes=30):
        self.presence = {}  # {id: [timestamps]}
        self.required_duration = timedelta(minutes=required_minutes)

    def update_presence(self, student_id):
        now = datetime.now()
        if student_id not in self.presence:
            self.presence[student_id] = [now]
        else:
            last_seen = self.presence[student_id][-1]
            if (now - last_seen).seconds >= 60:
                self.presence[student_id].append(now)

    def has_attended(self, student_id):
        if student_id not in self.presence:
            return False

        times = self.presence[student_id]
        if len(times) < 2:
            return False

        total = timedelta()
        for i in range(1, len(times)):
            diff = times[i] - times[i-1]
            if diff.seconds <= 180:
                total += diff

        return total >= self.required_duration
