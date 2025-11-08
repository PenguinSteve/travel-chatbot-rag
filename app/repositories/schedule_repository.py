from pymongo.collection import Collection
from app.models.schedule_schema import ScheduleItem
from datetime import datetime
class ScheduleRepository:
    def __init__(self, db):
        self.collection: Collection = db["schedules"]

    def create_schedule(self, schedule_item):
        if hasattr(schedule_item, "model_dump"):
            schedule_item = schedule_item.model_dump()

        elif isinstance(schedule_item, str):
            import json
            schedule_item = json.loads(schedule_item)

        self.collection.insert_one({
            "user_id": schedule_item["user_id"],
            "trip_id": schedule_item["trip_id"],
            "location": schedule_item["location"],
            "duration_days": schedule_item["duration_days"],
            "start_date": schedule_item["start_date"],
            "end_date": schedule_item["end_date"],
            "weather_summary": schedule_item["weather_summary"],
            "itinerary": schedule_item["itinerary"],
            "accommodation": schedule_item["accommodation"],
            "tips": schedule_item["tips"],
            "created_at": schedule_item.get("created_at", datetime.now()) ,
        })


    def update_schedule(self, trip_id: str, schedule_item: ScheduleItem):
        self.collection.update_one(
            {"trip_id": trip_id},
            {
                "$set": {
                    "location": schedule_item["location"],
                    "duration_days": schedule_item["duration_days"],
                    "start_date": schedule_item["start_date"],
                    "end_date": schedule_item["end_date"],
                    "weather_summary": schedule_item["weather_summary"],
                    "itinerary": schedule_item["itinerary"],
                    "accommodation": schedule_item["accommodation"],
                    "tips": schedule_item["tips"],
                }
            },
            upsert=True
        )