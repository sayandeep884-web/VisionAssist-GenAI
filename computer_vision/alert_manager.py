import time


class AlertManager:

    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.last_alert = None
        self.last_time = 0

    def should_alert(self, navigation):

        current_time = time.time()

        direction = navigation["direction"]
        urgency = navigation["urgency"]
        object_name = navigation.get("closest_object")

        current_alert = (
            object_name,
            direction,
            urgency
        )

        # First alert
        if self.last_alert is None:
            self.last_alert = current_alert
            self.last_time = current_time
            return True

        # Direction/object/urgency changed
        if current_alert != self.last_alert:

            self.last_alert = current_alert
            self.last_time = current_time
            return True

        # High urgency should be repeated
        if urgency == "high":
            if current_time - self.last_time >= 1.5:
                self.last_time = current_time
                return True

        # Normal alerts use cooldown
        if current_time - self.last_time >= self.cooldown:
            self.last_time = current_time
            return True

        return False