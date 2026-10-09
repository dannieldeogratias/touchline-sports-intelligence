from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import Base, SessionLocal, engine, get_session
from .models import Event, EventRevision, Match, Organization, ProcessingJob, VideoAsset
from .schemas import (
    EventCreate,
    EventRead,
    EventRevisionRead,
    EventUpdate,
    MatchCreate,
    MatchRead,
    MatchUpdate,
    OrganizationCreate,
    OrganizationRead,
    ProcessingJobCreate,
    ProcessingJobRead,
    VideoCreate,
    VideoRead,
)


async def run_detection_job(job_id: str) -> None:
    async with SessionLocal() as session:
        job = await session.get(ProcessingJob, job_id)
        if job is None:
            return
        job.status = "RUNNING"
        job.progress = 15
        await session.commit()
        if job.mode == "YOLO":
            try:
                from .vision import run_yolo_detection

                video = await session.get(VideoAsset, job.video_id)
                if video is None or video.storage_key is None:
                    raise RuntimeError("The source video is unavailable")
                summary = await run_yolo_detection(
                    settings.media_root / video.storage_key,
                    settings.yolo_model_path,
                    settings.analysis_root / f"{job.id}.json",
                    settings.analysis_root / f"{job.id}-tracked.mp4",
                    settings.ball_model_path,
                )
            except Exception as error:
                job.status = "FAILED"
                job.error_message = str(error) or repr(error) or "Vision worker failed without an error message"
                await session.commit()
                return
        else:
            summary = {"frames_sampled": 120, "detections": {"player": 184, "ball": 37, "referee": 12}}
        job.progress = 100
        job.status = "SUCCEEDED"
        job.result_summary = json.dumps(summary)
        await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Session = Annotated[AsyncSession, Depends(get_session)]
TenantId = Annotated[str, Header(alias="X-Organization-ID")]


async def tenant_match(session: AsyncSession, match_id: str, organization_id: str) -> Match:
    match = await session.scalar(
        select(Match).where(Match.id == match_id, Match.organization_id == organization_id)
    )
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


def event_snapshot(event: Event, changed_by: str) -> EventRevision:
    return EventRevision(
        event_id=event.id,
        organization_id=event.organization_id,
        revision_number=event.revision_number,
        timestamp_ms=event.timestamp_ms,
        event_type=event.event_type,
        team_name=event.team_name,
        player_name=event.player_name,
        pitch_x=event.pitch_x,
        pitch_y=event.pitch_y,
        source=event.source,
        verification_status=event.verification_status,
        note=event.note,
        changed_by=changed_by,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/organizations", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
async def create_organization(payload: OrganizationCreate, session: Session) -> Organization:
    if await session.scalar(select(Organization).where(Organization.slug == payload.slug)):
        raise HTTPException(status_code=409, detail="Organization slug already exists")
    organization = Organization(**payload.model_dump())
    session.add(organization)
    await session.commit()
    await session.refresh(organization)
    return organization


@app.get("/api/v1/organizations/by-slug/{slug}", response_model=OrganizationRead)
async def get_organization_by_slug(slug: str, session: Session) -> Organization:
    organization = await session.scalar(select(Organization).where(Organization.slug == slug))
    if organization is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return organization


@app.post("/api/v1/matches", response_model=MatchRead, status_code=status.HTTP_201_CREATED)
async def create_match(payload: MatchCreate, organization_id: TenantId, session: Session) -> Match:
    if await session.get(Organization, organization_id) is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    match = Match(organization_id=organization_id, **payload.model_dump())
    session.add(match)
    await session.commit()
    await session.refresh(match)
    return match


@app.get("/api/v1/matches", response_model=list[MatchRead])
async def list_matches(organization_id: TenantId, session: Session) -> list[Match]:
    result = await session.scalars(
        select(Match).where(Match.organization_id == organization_id).order_by(Match.created_at.desc())
    )
    return list(result)


@app.get("/api/v1/matches/{match_id}", response_model=MatchRead)
async def get_match(match_id: str, organization_id: TenantId, session: Session) -> Match:
    return await tenant_match(session, match_id, organization_id)


@app.patch("/api/v1/matches/{match_id}", response_model=MatchRead)
async def update_match(
    match_id: str, payload: MatchUpdate, organization_id: TenantId, session: Session
) -> Match:
    match = await tenant_match(session, match_id, organization_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(match, field, value)
    await session.commit()
    await session.refresh(match)
    return match


@app.post("/api/v1/matches/{match_id}/videos", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def create_video(match_id: str, payload: VideoCreate, organization_id: TenantId, session: Session) -> VideoAsset:
    await tenant_match(session, match_id, organization_id)
    video = VideoAsset(organization_id=organization_id, match_id=match_id, **payload.model_dump())
    session.add(video)
    await session.commit()
    await session.refresh(video)
    return video


@app.post("/api/v1/matches/{match_id}/media", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def upload_video(
    match_id: str,
    organization_id: TenantId,
    session: Session,
    file: UploadFile = File(),
    duration_ms: int = Form(),
    source_start_ms: int = Form(default=0),
) -> VideoAsset:
    await tenant_match(session, match_id, organization_id)
    if duration_ms <= 0 or duration_ms > settings.max_development_duration_ms:
        raise HTTPException(status_code=422, detail="Development uploads must be no longer than 10 minutes")
    if file.content_type not in {"video/mp4", "video/quicktime", "video/webm"}:
        raise HTTPException(status_code=415, detail="Supported formats are MP4, MOV and WebM")

    suffix = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}[file.content_type]
    video = VideoAsset(
        organization_id=organization_id,
        match_id=match_id,
        filename=Path(file.filename or f"upload{suffix}").name,
        content_type=file.content_type,
        duration_ms=duration_ms,
        source_start_ms=source_start_ms,
        rights_status="development_only",
    )
    session.add(video)
    await session.flush()
    relative_path = Path(organization_id) / match_id / f"{video.id}{suffix}"
    destination = settings.media_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    first_chunk = await file.read(1024 * 1024)
    valid_signature = (file.content_type == "video/webm" and first_chunk.startswith(b"\x1aE\xdf\xa3")) or (
        file.content_type != "video/webm" and len(first_chunk) >= 12 and first_chunk[4:8] == b"ftyp"
    )
    if not valid_signature:
        raise HTTPException(status_code=422, detail="File signature does not match the selected video format")
    try:
        with destination.open("wb") as output:
            chunk = first_chunk
            while chunk:
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Video exceeds the local upload limit")
                digest.update(chunk)
                output.write(chunk)
                chunk = await file.read(1024 * 1024)
        video.checksum_sha256 = digest.hexdigest()
        video.storage_key = relative_path.as_posix()
        await session.commit()
        await session.refresh(video)
        return video
    except Exception:
        destination.unlink(missing_ok=True)
        await session.rollback()
        raise


@app.post("/api/v1/matches/{match_id}/jobs", response_model=ProcessingJobRead, status_code=status.HTTP_202_ACCEPTED)
async def create_processing_job(
    match_id: str,
    payload: ProcessingJobCreate,
    background_tasks: BackgroundTasks,
    organization_id: TenantId,
    session: Session,
) -> ProcessingJob:
    await tenant_match(session, match_id, organization_id)
    video = await session.scalar(select(VideoAsset).where(
        VideoAsset.id == payload.video_id,
        VideoAsset.match_id == match_id,
        VideoAsset.organization_id == organization_id,
    ))
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    job = ProcessingJob(
        organization_id=organization_id, match_id=match_id, video_id=video.id, mode=payload.mode
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background_tasks.add_task(run_detection_job, job.id)
    return job


@app.get("/api/v1/jobs/{job_id}", response_model=ProcessingJobRead)
async def get_processing_job(job_id: str, organization_id: TenantId, session: Session) -> ProcessingJob:
    job = await session.scalar(select(ProcessingJob).where(
        ProcessingJob.id == job_id, ProcessingJob.organization_id == organization_id
    ))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/jobs/{job_id}/analysis")
async def get_job_analysis(job_id: str, organization_id: TenantId, session: Session) -> FileResponse:
    job = await session.scalar(select(ProcessingJob).where(
        ProcessingJob.id == job_id, ProcessingJob.organization_id == organization_id
    ))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="Analysis is not ready")
    output_path = settings.analysis_root / f"{job.id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Analysis artifact not found")
    return FileResponse(output_path, media_type="application/json", filename=f"analysis-{job.id}.json")


@app.get("/api/v1/jobs/{job_id}/tracked-video")
async def get_tracked_video(job_id: str, organization_id: TenantId, session: Session) -> FileResponse:
    job = await session.scalar(select(ProcessingJob).where(
        ProcessingJob.id == job_id, ProcessingJob.organization_id == organization_id
    ))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="Tracked video is not ready")
    video_path = settings.analysis_root / f"{job.id}-tracked.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Tracked video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"tracked-{job.id}.mp4")


@app.post("/api/v1/matches/{match_id}/events", response_model=EventRead, status_code=status.HTTP_201_CREATED)
async def create_event(match_id: str, payload: EventCreate, organization_id: TenantId, session: Session) -> Event:
    await tenant_match(session, match_id, organization_id)
    event = Event(organization_id=organization_id, match_id=match_id, **payload.model_dump())
    session.add(event)
    await session.flush()
    session.add(event_snapshot(event, "local-analyst"))
    await session.commit()
    await session.refresh(event)
    return event


@app.get("/api/v1/matches/{match_id}/events", response_model=list[EventRead])
async def list_events(
    match_id: str,
    organization_id: TenantId,
    session: Session,
    event_type: str | None = Query(default=None),
    include_rejected: bool = Query(default=False),
) -> list[Event]:
    await tenant_match(session, match_id, organization_id)
    statement = select(Event).where(Event.match_id == match_id, Event.organization_id == organization_id)
    if event_type:
        statement = statement.where(Event.event_type == event_type)
    if not include_rejected:
        statement = statement.where(Event.verification_status != "REJECTED")
    result = await session.scalars(statement.order_by(Event.timestamp_ms))
    return list(result)


@app.patch("/api/v1/events/{event_id}", response_model=EventRead)
async def update_event(event_id: str, payload: EventUpdate, organization_id: TenantId, session: Session) -> Event:
    event = await session.scalar(
        select(Event).where(Event.id == event_id, Event.organization_id == organization_id)
    )
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    changes = payload.model_dump(exclude_unset=True)
    changed_by = changes.pop("changed_by", "local-analyst")
    for field, value in changes.items():
        setattr(event, field, value)
    event.revision_number += 1
    session.add(event_snapshot(event, changed_by))
    await session.commit()
    await session.refresh(event)
    return event


@app.get("/api/v1/events/{event_id}/revisions", response_model=list[EventRevisionRead])
async def list_event_revisions(event_id: str, organization_id: TenantId, session: Session) -> list[EventRevision]:
    event = await session.scalar(
        select(Event).where(Event.id == event_id, Event.organization_id == organization_id)
    )
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    revisions = await session.scalars(
        select(EventRevision)
        .where(EventRevision.event_id == event_id, EventRevision.organization_id == organization_id)
        .order_by(EventRevision.revision_number)
    )
    return list(revisions)
