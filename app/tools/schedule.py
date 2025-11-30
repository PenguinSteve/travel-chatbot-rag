from app.repositories.schedule_repository import ScheduleRepository
from app.config.mongodb import get_database_schedule

def schedule_trip(trip_details):
    db = get_database_schedule()
    schedule_repo = ScheduleRepository(db)
    schedule_repo.create_schedule(trip_details)
    print("Schedule created successfully.")

