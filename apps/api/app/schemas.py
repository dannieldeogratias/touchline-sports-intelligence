from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)


class OrganizationRead(ORMModel):
    id: str
    name: str
    slug: str
    created_at: datetime


class MatchCreate(BaseModel):
    home_team: str = Field(min_length=1, max_length=120)
    away_team: str = Field(min_length=1, max_length=120)
    competition: str | None = Field(default=None, max_length=120)
    kickoff_at: datetime | None = None


class MatchUpdate(BaseModel):
    home_team: str | None = Field(default=None, min_length=1, max_length=120)
    away_team: str | None = Field(default=None, min_length=1, max_length=120)
    competition: str | None = Field(default=None, max_length=120)


class MatchRead(ORMModel):
    id: str
    organization_id: str
    home_team: str
    away_team: str
    competition: str | None
    kickoff_at: datetime | None
    created_at: datetime


class VideoCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    duration_ms: int = Field(gt=0, le=7_200_000)
    source_start_ms: int = Field(default=0, ge=0)
    checksum_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    rights_status: Literal["development_only", "authorized", "restricted"] = "development_only"
    storage_key: str | None = Field(default=None, max_length=500)


class VideoRead(ORMModel):
    id: str
    organization_id: str
    match_id: str
    filename: str
    content_type: str
    duration_ms: int
    source_start_ms: int
    checksum_sha256: str | None
    rights_status: str
    storage_key: str | None
    created_at: datetime


EventSource = Literal["AI", "AI_SUGGESTION", "ANALYST", "AI_CORRECTED"]
VerificationStatus = Literal["UNVERIFIED", "CONFIRMED", "CORRECTED", "REJECTED"]


class EventCreate(BaseModel):
    timestamp_ms: int = Field(ge=0)
    event_type: str = Field(min_length=1, max_length=50)
    team_name: str | None = Field(default=None, max_length=120)
    player_name: str | None = Field(default=None, max_length=120)
    pitch_x: float | None = Field(default=None, ge=0, le=100)
    pitch_y: float | None = Field(default=None, ge=0, le=100)
    source: EventSource = "ANALYST"
    verification_status: VerificationStatus = "UNVERIFIED"
    note: str | None = None


class EventUpdate(BaseModel):
    timestamp_ms: int | None = Field(default=None, ge=0)
    event_type: str | None = Field(default=None, min_length=1, max_length=50)
    team_name: str | None = Field(default=None, max_length=120)
    player_name: str | None = Field(default=None, max_length=120)
    pitch_x: float | None = Field(default=None, ge=0, le=100)
    pitch_y: float | None = Field(default=None, ge=0, le=100)
    source: EventSource | None = None
    verification_status: VerificationStatus | None = None
    note: str | None = None
    changed_by: str = Field(default="local-analyst", max_length=120)


class EventRead(ORMModel):
    id: str
    organization_id: str
    match_id: str
    timestamp_ms: int
    event_type: str
    team_name: str | None
    player_name: str | None
    pitch_x: float | None
    pitch_y: float | None
    source: str
    verification_status: str
    note: str | None
    revision_number: int
    created_at: datetime
    updated_at: datetime


class EventRevisionRead(ORMModel):
    id: str
    event_id: str
    revision_number: int
    timestamp_ms: int
    event_type: str
    team_name: str | None
    player_name: str | None
    source: str
    verification_status: str
    note: str | None
    changed_by: str
    created_at: datetime


class ProcessingJobCreate(BaseModel):
    video_id: str
    mode: Literal["MOCK", "YOLO"] = "MOCK"


class ProcessingJobRead(ORMModel):
    id: str
    organization_id: str
    match_id: str
    video_id: str
    kind: str
    mode: str
    status: str
    progress: int
    result_summary: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
