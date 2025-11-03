from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class WeatherSummary(BaseModel):
    avg_temp: Optional[float] = None
    condition: Optional[str] = None
    notes: Optional[str] = None


class Activity(BaseModel):
    time_start: str
    time_end: str
    description: str
    type: Literal["Food", "Attraction", "Accommodation", "Festival", "Transport"]


class Itinerary(BaseModel):
    day: int
    title: str
    activities: List[Activity]


class Accommodation(BaseModel): 
    name: str
    address: str
    price_range: str
    notes: Optional[str] = None


class ScheduleItem(BaseModel):
    user_id: str
    trip_id: str
    location: str
    duration_days: int
    start_date: datetime
    end_date: datetime
    weather_summary: WeatherSummary
    itinerary: List[Itinerary]
    accommodation: Accommodation
    tips: List[str]
    created_at: datetime = Field(default_factory=datetime.now) 

