from app.models.schedule_schema import ScheduleItem
from app.repositories.schedule_repository import ScheduleRepository
from app.config.mongodb import get_database

def schedule_trip(trip_details: ScheduleItem):
    db = get_database()
    schedule_repo = ScheduleRepository(db)
    print('Creating schedule for trip:', trip_details.trip_id)
    schedule_repo.create_schedule(trip_details)
    print('Schedule created successfully.')

if __name__ == "__main__":
    sample_trip = ScheduleItem(
        user_id="123456",
        trip_id="trip123",
        location="Paris",
        duration_days=5,
        start_date="2024-09-01T00:00:00Z",
        end_date="2024-09-06T00:00:00Z",
        weather_summary={
            "avg_temp": 20.5,
            "condition": "Sunny",
            "notes": "Perfect weather for sightseeing."
        },
        itinerary=[
            {
                "day": 1,
                "title": "Arrival and City Tour",
                "activities": [
                    {
                        "time_start": "10:00",
                        "time_end": "12:00",
                        "description": "Visit the Eiffel Tower",
                        "type": "Attraction"
                    },
                    {
                        "time_start": "13:00",
                        "time_end": "14:00",
                        "description": "Lunch at local cafe",
                        "type": "Food"
                    }
                ]
            }
        ],
        accommodation={
            "address": "123 Paris St, Paris, France",
            "price_range": "500000",
            "notes": "Close to major attractions."
        },
        tips=["Buy tickets in advance", "Use public transport"]
    )
    schedule_trip(sample_trip)
