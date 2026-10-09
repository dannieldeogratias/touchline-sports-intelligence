from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (Index("ix_matches_org_created", "organization_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(120), nullable=False)
    away_team: Mapped[str] = mapped_column(String(120), nullable=False)
    competition: Mapped[str | None] = mapped_column(String(120))
    kickoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    videos: Mapped[list["VideoAsset"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class VideoAsset(Base):
    __tablename__ = "video_assets"
    __table_args__ = (
        CheckConstraint("duration_ms > 0", name="ck_video_duration_positive"),
        Index("ix_video_assets_org_match", "organization_id", "match_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    source_start_ms: Mapped[int] = mapped_column(Integer, default=0)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    rights_status: Mapped[str] = mapped_column(String(40), default="development_only")
    storage_key: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    match: Mapped[Match] = relationship(back_populates="videos")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint("timestamp_ms >= 0", name="ck_event_timestamp_nonnegative"),
        CheckConstraint("pitch_x IS NULL OR (pitch_x >= 0 AND pitch_x <= 100)", name="ck_event_pitch_x"),
        CheckConstraint("pitch_y IS NULL OR (pitch_y >= 0 AND pitch_y <= 100)", name="ck_event_pitch_y"),
        Index("ix_events_org_match_time", "organization_id", "match_id", "timestamp_ms"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    team_name: Mapped[str | None] = mapped_column(String(120))
    player_name: Mapped[str | None] = mapped_column(String(120))
    pitch_x: Mapped[float | None] = mapped_column(Float)
    pitch_y: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), default="ANALYST")
    verification_status: Mapped[str] = mapped_column(String(30), default="UNVERIFIED")
    note: Mapped[str | None] = mapped_column(Text)
    revision_number: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    match: Mapped[Match] = relationship(back_populates="events")
    revisions: Mapped[list["EventRevision"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", order_by="EventRevision.revision_number"
    )


class EventRevision(Base):
    __tablename__ = "event_revisions"
    __table_args__ = (
        Index("ix_event_revisions_event_revision", "event_id", "revision_number", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    team_name: Mapped[str | None] = mapped_column(String(120))
    player_name: Mapped[str | None] = mapped_column(String(120))
    pitch_x: Mapped[float | None] = mapped_column(Float)
    pitch_y: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(30), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    changed_by: Mapped[str] = mapped_column(String(120), default="local-analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    event: Mapped[Event] = relationship(back_populates="revisions")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (Index("ix_processing_jobs_org_match", "organization_id", "match_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_assets.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), default="OBJECT_DETECTION")
    mode: Mapped[str] = mapped_column(String(20), default="MOCK")
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    result_summary: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
