import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeft,
  ArrowsOutSimple,
  ArrowUpRight,
  Broadcast,
  CaretDown,
  ChartBar,
  Check,
  Cpu,
  CursorClick,
  DownloadSimple,
  FastForward,
  Flag,
  FolderOpen,
  LineSegment,
  MagnifyingGlass,
  MapPin,
  Pause,
  Play,
  Plus,
  Selection,
  SoccerBall,
  Timer,
  Trash,
  UploadSimple,
  UserCircle,
  UsersThree,
  VideoCamera,
  X,
} from '@phosphor-icons/react'
import './App.css'
import {
  bootstrapWorkspace, createApiEvent, createProcessingJob, getJobAnalysis, getProcessingJob, getTrackedVideo,
  updateApiEvent, updateMatch, uploadApiVideo, type AnalysisData, type ApiEvent,
} from './api'

type EventKind = 'Pass' | 'Shot' | 'Recovery' | 'Press' | 'Note'
type MatchEvent = {
  id: string
  time: number
  kind: EventKind
  team: string
  player: string
  status: 'confirmed' | 'review'
  note: string
}

const initialEvents: MatchEvent[] = [
  { id: 'sample-1', time: 38, kind: 'Recovery', team: 'Home team', player: 'Unassigned', status: 'confirmed', note: 'Regain in the middle third' },
  { id: 'sample-2', time: 74, kind: 'Pass', team: 'Home team', player: 'Unassigned', status: 'confirmed', note: 'Progressive pass into left half-space' },
]

const fromApiEvent = (event: ApiEvent): MatchEvent => ({
  id: event.id,
  time: event.timestamp_ms / 1000,
  kind: `${event.event_type.charAt(0)}${event.event_type.slice(1).toLowerCase()}` as EventKind,
  team: event.team_name ?? 'Unassigned team',
  player: event.player_name ?? 'Unassigned',
  status: event.verification_status === 'CONFIRMED' ? 'confirmed' : 'review',
  note: event.note ?? '',
})

const formatTime = (seconds: number) => {
  const safe = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`
}
const teamCode = (name: string) => name.split(/\s+/).filter(Boolean).map((part) => part[0]).join('').slice(0, 3).toUpperCase() || 'TEAM'
type Point = { x: number; y: number }
type PitchRegion = 'full' | 'left' | 'centre' | 'right'
type Calibration = { time: number; source: Point[]; target: Point[]; region: PitchRegion }
const pitchRegions: Record<PitchRegion, { label: string; target: Point[] }> = {
  full: { label: 'Full pitch', target: [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }] },
  left: { label: 'Left half', target: [{ x: 0, y: 0 }, { x: 50, y: 0 }, { x: 50, y: 100 }, { x: 0, y: 100 }] },
  centre: { label: 'Centre half', target: [{ x: 25, y: 0 }, { x: 75, y: 0 }, { x: 75, y: 100 }, { x: 25, y: 100 }] },
  right: { label: 'Right half', target: [{ x: 50, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 50, y: 100 }] },
}

const solveLinear = (matrix: number[][], values: number[]) => {
  const rows = matrix.map((row, index) => [...row, values[index]])
  for (let column = 0; column < values.length; column += 1) {
    let pivot = column
    for (let row = column + 1; row < rows.length; row += 1) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row
    ;[rows[column], rows[pivot]] = [rows[pivot], rows[column]]
    const divisor = rows[column][column]
    if (Math.abs(divisor) < 1e-9) return null
    for (let item = column; item <= values.length; item += 1) rows[column][item] /= divisor
    for (let row = 0; row < rows.length; row += 1) {
      if (row === column) continue
      const factor = rows[row][column]
      for (let item = column; item <= values.length; item += 1) rows[row][item] -= factor * rows[column][item]
    }
  }
  return rows.map((row) => row[values.length])
}

const makeHomography = (source: Point[], target = pitchRegions.full.target) => {
  if (source.length !== 4) return null
  const matrix: number[][] = [], values: number[] = []
  source.forEach((point, index) => {
    const out = target[index]
    matrix.push([point.x, point.y, 1, 0, 0, 0, -out.x * point.x, -out.x * point.y]); values.push(out.x)
    matrix.push([0, 0, 0, point.x, point.y, 1, -out.y * point.x, -out.y * point.y]); values.push(out.y)
  })
  return solveLinear(matrix, values)
}

const calibrationAt = (items: Calibration[], time: number, cuts: number[] = []) => {
  const segmentStart = [...cuts].reverse().find((cut) => cut <= time) ?? 0
  return [...items].reverse().find((item) => item.time >= segmentStart && item.time <= time + .25) ?? null
}

const projectPoint = (matrix: number[] | null, point: Point) => {
  if (!matrix) return null
  const denominator = matrix[6] * point.x + matrix[7] * point.y + 1
  if (Math.abs(denominator) < 1e-9) return null
  return { x: (matrix[0] * point.x + matrix[1] * point.y + matrix[2]) / denominator,
    y: (matrix[3] * point.x + matrix[4] * point.y + matrix[5]) / denominator }
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const pendingFileRef = useRef<File | null>(null)
  const uploadStartedRef = useRef(false)
  const [videoUrl, setVideoUrl] = useState('')
  const [fileName, setFileName] = useState('10-minute development segment')
  const [duration, setDuration] = useState(600)
  const [currentTime, setCurrentTime] = useState(198)
  const [playing, setPlaying] = useState(false)
  const [view, setView] = useState<'video' | 'pitch' | 'analysis'>('video')
  const [activeTool, setActiveTool] = useState('select')
  const [query, setQuery] = useState('')
  const [selectedEvent, setSelectedEvent] = useState<string>('sample-1')
  const [connection, setConnection] = useState<'connecting' | 'connected' | 'offline'>('connecting')
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'error'>('saved')
  const [workspaceIds, setWorkspaceIds] = useState<{ organizationId: string; matchId: string } | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [videoAssetId, setVideoAssetId] = useState('')
  const [mediaState, setMediaState] = useState<'idle' | 'uploading' | 'ready' | 'processing' | 'done' | 'error'>('idle')
  const [processingSummary, setProcessingSummary] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null)
  const [trackedVideoUrl, setTrackedVideoUrl] = useState('')
  const [videoSource, setVideoSource] = useState<'original' | 'tracked'>('original')
  const [homeTeam, setHomeTeam] = useState('Home team')
  const [awayTeam, setAwayTeam] = useState('Away team')
  const [homeDraft, setHomeDraft] = useState('')
  const [awayDraft, setAwayDraft] = useState('')
  const [editingTeams, setEditingTeams] = useState(false)
  const [calibrationMode, setCalibrationMode] = useState(false)
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([])
  const [calibrationRegion, setCalibrationRegion] = useState<PitchRegion>('full')
  const [calibrations, setCalibrations] = useState<Calibration[]>(() => {
    const saved = localStorage.getItem('touchline-calibrations')
    return saved ? JSON.parse(saved) : []
  })
  const [events, setEvents] = useState<MatchEvent[]>(() => {
    const saved = localStorage.getItem('touchline-events')
    return saved ? JSON.parse(saved) : initialEvents
  })

  useEffect(() => {
    let cancelled = false
    bootstrapWorkspace()
      .then(({ organization, match, events: apiEvents }) => {
        if (cancelled) return
        setWorkspaceIds({ organizationId: organization.id, matchId: match.id })
        setHomeTeam(match.home_team)
        setAwayTeam(match.away_team)
        setEvents(apiEvents.map(fromApiEvent))
        setSelectedEvent(apiEvents[0]?.id ?? '')
        setConnection('connected')
      })
      .catch(() => {
        if (cancelled) return
        const saved = localStorage.getItem('touchline-events')
        if (saved) setEvents(JSON.parse(saved))
        setConnection('offline')
      })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    localStorage.setItem('touchline-events', JSON.stringify(events))
  }, [events])
  useEffect(() => {
    localStorage.setItem('touchline-calibrations', JSON.stringify(calibrations))
  }, [calibrations])

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
  }, [videoUrl])

  useEffect(() => () => {
    if (trackedVideoUrl) URL.revokeObjectURL(trackedVideoUrl)
  }, [trackedVideoUrl])

  const filteredEvents = useMemo(() => {
    const term = query.toLowerCase()
    return events.filter((event) =>
      [event.kind, event.player, event.team, event.note].some((value) => value.toLowerCase().includes(term)),
    )
  }, [events, query])

  const activeEvent = events.find((event) => event.id === selectedEvent)
  const activeFrame = useMemo(() => {
    if (!analysis?.frames.length) return null
    return analysis.frames.reduce((nearest, frame) =>
      Math.abs(frame.t - currentTime) < Math.abs(nearest.t - currentTime) ? frame : nearest,
    )
  }, [analysis, currentTime])
  const cameraCuts = useMemo(() => analysis?.summary.camera_cuts ?? [], [analysis])
  const activeCalibration = useMemo(() => calibrationAt(calibrations, currentTime, cameraCuts), [calibrations, currentTime, cameraCuts])
  const homography = useMemo(() => activeCalibration ? makeHomography(activeCalibration.source, activeCalibration.target) : null, [activeCalibration])
  const livePositions = useMemo(() => (activeFrame?.d ?? []).flatMap((detection) => {
    if (detection.c !== 'player' || detection.id === undefined) return []
    const point = projectPoint(homography, { x: (detection.b[0] + detection.b[2]) / 2, y: detection.b[3] })
    return point && point.x >= -5 && point.x <= 105 && point.y >= -5 && point.y <= 105
      ? [{ ...point, id: detection.id, team: detection.team ?? 0 }] : []
  }), [activeFrame, homography])
  const liveBall = useMemo(() => {
    const ball = activeFrame?.d.find((detection) => detection.c === 'ball')
    return ball ? projectPoint(homography, { x: (ball.b[0] + ball.b[2]) / 2, y: ball.b[3] }) : null
  }, [activeFrame, homography])
  const visibleTeamCounts = useMemo(() => ({
    home: livePositions.filter((position) => position.team === 0).length,
    away: livePositions.filter((position) => position.team === 1).length,
  }), [livePositions])
  const pitchTrails = useMemo(() => {
    if (!analysis || !activeCalibration) return new Map<number, Point[]>()
    const trails = new Map<number, Point[]>()
    analysis.frames.filter((frame) => frame.t <= currentTime && frame.t >= currentTime - 4).forEach((frame) => {
      const frameCalibration = calibrationAt(calibrations, frame.t, cameraCuts)
      const frameHomography = frameCalibration ? makeHomography(frameCalibration.source, frameCalibration.target) : null
      frame.d.forEach((detection) => {
        if (detection.c !== 'player' || detection.id === undefined) return
        const point = projectPoint(frameHomography, { x: (detection.b[0] + detection.b[2]) / 2, y: detection.b[3] })
        if (point && point.x >= 0 && point.x <= 100 && point.y >= 0 && point.y <= 100)
          trails.set(detection.id, [...(trails.get(detection.id) ?? []), point])
      })
    })
    return trails
  }, [analysis, activeCalibration, calibrations, currentTime, cameraCuts])

  const seek = (time: number) => {
    const next = Math.min(time, duration)
    setCurrentTime(next)
    if (videoRef.current) videoRef.current.currentTime = next
  }

  const togglePlay = () => {
    if (!videoRef.current) {
      setPlaying((value) => !value)
      return
    }
    if (videoRef.current.paused) videoRef.current.play()
    else videoRef.current.pause()
  }

  const addEvent = async (kind: EventKind) => {
    const newId = `local-${crypto.randomUUID()}`
    const newEvent: MatchEvent = {
      id: newId,
      time: currentTime,
      kind,
      team: homeTeam,
      player: 'Unassigned',
      status: 'review',
      note: `New ${kind.toLowerCase()} at ${formatTime(currentTime)}`,
    }
    setEvents((items) => [...items, newEvent].sort((a, b) => a.time - b.time))
    setSelectedEvent(newEvent.id)
    if (!workspaceIds) return
    setSaveState('saving')
    try {
      const saved = await createApiEvent(workspaceIds.organizationId, workspaceIds.matchId, {
        timestamp_ms: Math.round(newEvent.time * 1000), event_type: kind.toUpperCase(),
        team_name: newEvent.team, player_name: newEvent.player, source: 'ANALYST',
        verification_status: 'UNVERIFIED', note: newEvent.note,
      })
      const mapped = fromApiEvent(saved)
      setEvents((items) => items.map((item) => item.id === newId ? mapped : item))
      setSelectedEvent(mapped.id)
      setSaveState('saved')
    } catch {
      setSaveState('error')
      setConnection('offline')
    }
  }

  const confirmSelected = async () => {
    setEvents((items) => items.map((event) =>
      event.id === selectedEvent ? { ...event, status: 'confirmed' } : event,
    ))
    if (!workspaceIds || selectedEvent.startsWith('local-') || selectedEvent.startsWith('sample-')) return
    setSaveState('saving')
    try {
      await updateApiEvent(workspaceIds.organizationId, selectedEvent, { verification_status: 'CONFIRMED' })
      setSaveState('saved')
    } catch { setSaveState('error') }
  }

  const deleteSelected = async () => {
    const eventId = selectedEvent
    setEvents((items) => items.filter((event) => event.id !== selectedEvent))
    setSelectedEvent('')
    if (!workspaceIds || eventId.startsWith('local-') || eventId.startsWith('sample-')) return
    setSaveState('saving')
    try {
      await updateApiEvent(workspaceIds.organizationId, eventId, { verification_status: 'REJECTED' })
      setSaveState('saved')
    } catch { setSaveState('error') }
  }

  const onVideoFile = (file?: File) => {
    if (!file) return
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    const nextUrl = URL.createObjectURL(file)
    setVideoUrl(nextUrl)
    setFileName(file.name)
    setPendingFile(file)
    pendingFileRef.current = file
    uploadStartedRef.current = false
    setVideoAssetId('')
    setMediaState('idle')
    setProcessingSummary('')
    setAnalysis(null)
    if (trackedVideoUrl) URL.revokeObjectURL(trackedVideoUrl)
    setTrackedVideoUrl('')
    setVideoSource('original')
    setCalibrationPoints([])
    setCalibrations([])
    setCalibrationMode(false)
    setCurrentTime(0)
    setPlaying(false)
  }

  const registerVideo = async (mediaDuration: number) => {
    setDuration(mediaDuration || 600)
    const file = pendingFileRef.current ?? pendingFile
    if (!file || !workspaceIds) return
    if (uploadStartedRef.current) return
    uploadStartedRef.current = true
    setMediaState('uploading')
    try {
      const asset = await uploadApiVideo(
        workspaceIds.organizationId, workspaceIds.matchId, file, Math.round(mediaDuration * 1000),
      )
      setVideoAssetId(asset.id)
      setMediaState('ready')
    } catch (error) {
      setMediaState('error')
      setProcessingSummary(error instanceof Error ? error.message : 'Upload failed')
    }
  }

  useEffect(() => {
    const file = pendingFileRef.current
    if (!workspaceIds || !file || duration <= 0 || uploadStartedRef.current) return
    uploadStartedRef.current = true
    setMediaState('uploading')
    uploadApiVideo(workspaceIds.organizationId, workspaceIds.matchId, file, Math.round(duration * 1000))
      .then((asset) => { setVideoAssetId(asset.id); setMediaState('ready') })
      .catch((error: unknown) => {
        uploadStartedRef.current = false
        setMediaState('error')
        setProcessingSummary(error instanceof Error ? error.message : 'Upload failed')
      })
  }, [workspaceIds, pendingFile, duration])

  const runDetector = async () => {
    if (!workspaceIds || !videoAssetId) return
    setMediaState('processing')
    setProcessingSummary('Sampling frames and checking the detection pipeline…')
    try {
      let job = await createProcessingJob(workspaceIds.organizationId, workspaceIds.matchId, videoAssetId)
      for (let attempt = 0; attempt < 900 && !['SUCCEEDED', 'FAILED'].includes(job.status); attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000))
        job = await getProcessingJob(workspaceIds.organizationId, job.id)
      }
      if (job.status === 'SUCCEEDED') {
        const result = JSON.parse(job.result_summary ?? '{}')
        const fullAnalysis = await getJobAnalysis(workspaceIds.organizationId, job.id)
        const trackedBlob = await getTrackedVideo(workspaceIds.organizationId, job.id)
        const trackedUrl = URL.createObjectURL(trackedBlob)
        setAnalysis(fullAnalysis)
        setTrackedVideoUrl(trackedUrl)
        const candidates: MatchEvent[] = fullAnalysis.events.map((candidate, index) => ({
          id: `ai-${job.id}-${index}`, time: candidate.t,
          kind: candidate.type === 'PASS_CANDIDATE' ? 'Pass' : 'Shot',
          team: candidate.team === 1 ? awayTeam : homeTeam,
          player: candidate.team === 0 ? homeTeam : candidate.team === 1 ? awayTeam : 'Unassigned',
          status: 'review',
          note: `${candidate.type === 'PASS_CANDIDATE' ? 'Possible same-team pass' : 'Possible shot'} · ${Math.round(candidate.confidence * 100)}% confidence`,
        }))
        setEvents((items) => [...items.filter((event) => !event.id.startsWith('ai-')), ...candidates].sort((a, b) => a.time - b.time))
        const counts = result.detections ?? {}
        setProcessingSummary(`${counts.player ?? 0} player detections · ${counts.ball ?? 0} ball detections · tracked locally`)
        setMediaState('done')
        setView('analysis')
      } else {
        setProcessingSummary(job.error_message ?? 'Processing did not complete')
        setMediaState('error')
      }
    } catch (error) {
      setProcessingSummary(error instanceof Error ? error.message : 'Processing failed')
      setMediaState('error')
    }
  }

  const saveTeams = async () => {
    const nextHome = homeDraft.trim()
    const nextAway = awayDraft.trim()
    if (!workspaceIds || !nextHome || !nextAway) return
    setSaveState('saving')
    try {
      const match = await updateMatch(workspaceIds.organizationId, workspaceIds.matchId, {
        home_team: nextHome, away_team: nextAway,
      })
      setHomeTeam(match.home_team)
      setAwayTeam(match.away_team)
      setEditingTeams(false)
      setSaveState('saved')
    } catch { setSaveState('error') }
  }

  const calibratePoint = (event: React.MouseEvent<HTMLElement>) => {
    if (!calibrationMode || view !== 'video') return
    const bounds = event.currentTarget.getBoundingClientRect()
    const point = { x: (event.clientX - bounds.left) / bounds.width, y: (event.clientY - bounds.top) / bounds.height }
    setCalibrationPoints((items) => {
      const next = [...items, point].slice(0, 4)
      if (next.length === 4) {
        const calibration = { time: currentTime, source: next, target: pitchRegions[calibrationRegion].target, region: calibrationRegion }
        setCalibrations((existing) => [...existing.filter((item) => Math.abs(item.time - currentTime) > 1), calibration]
          .sort((first, second) => first.time - second.time))
        setCalibrationMode(false)
      }
      return next
    })
  }

  const tools = [
    { id: 'select', label: 'Select', icon: CursorClick },
    { id: 'line', label: 'Movement line', icon: LineSegment },
    { id: 'zone', label: 'Zone', icon: Selection },
    { id: 'marker', label: 'Marker', icon: MapPin },
  ]

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><SoccerBall weight="fill" /></div>
          <strong>Touchline</strong>
        </div>
        <button className="icon-button back-button" aria-label="Back"><ArrowLeft /></button>
        <button className="match-heading" onClick={() => { setHomeDraft(homeTeam); setAwayDraft(awayTeam); setEditingTeams(true) }} title="Edit teams">
          <span className="context-label">Analysis</span>
          <strong>{homeTeam} vs {awayTeam}</strong>
        </button>
        <div className="top-actions">
          <span className={`save-state ${saveState === 'error' ? 'save-error' : ''}`}>
            <Check weight="bold" /> {saveState === 'saving' ? 'Saving…' : saveState === 'error' ? 'Save failed' : connection === 'connected' ? 'Saved to API' : 'Saved locally'}
          </span>
          {trackedVideoUrl ? <a className="secondary-button" href={trackedVideoUrl} download={`${teamCode(homeTeam)}-${teamCode(awayTeam)}-tracked.mp4`}><DownloadSimple /> Export tracked video</a> : <button className="secondary-button" disabled><DownloadSimple /> Export</button>}
          <button className="avatar-button" aria-label="Open account"><UserCircle weight="fill" /></button>
        </div>
      </header>

      {editingTeams && <div className="team-editor-backdrop" role="presentation">
        <section className="team-editor" role="dialog" aria-modal="true" aria-label="Edit match teams">
          <div><span>Match setup</span><button onClick={() => setEditingTeams(false)} aria-label="Close"><X /></button></div>
          <label>Home team<input value={homeDraft} onChange={(event) => setHomeDraft(event.target.value)} autoFocus /></label>
          <label>Away team<input value={awayDraft} onChange={(event) => setAwayDraft(event.target.value)} /></label>
          <button className="confirm-button" onClick={saveTeams} disabled={!homeDraft.trim() || !awayDraft.trim()}><Check /> Save teams</button>
        </section>
      </div>}

      <section className="status-strip">
        <div><Broadcast weight="fill" /><span>Workspace</span><strong>Development match 01</strong></div>
        <div><Timer /><span>Segment</span><strong>10:00</strong></div>
        <div><SoccerBall /><span>Events</span><strong>{events.length}</strong></div>
        <div className="model-status"><Cpu /><span>API</span><strong>{connection === 'connected' ? 'Connected' : connection === 'connecting' ? 'Connecting' : 'Local fallback'}</strong></div>
      </section>

      <section className="workspace">
        <aside className="left-rail" aria-label="Annotation tools">
          {tools.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={activeTool === id ? 'tool-button active' : 'tool-button'}
              onClick={() => setActiveTool(id)}
              aria-label={label}
              title={label}
            >
              <Icon />
            </button>
          ))}
          <div className="tool-divider" />
          <button className="tool-button" aria-label="Delete selected annotation" title="Delete"><Trash /></button>
        </aside>

        <section className="stage">
          <div className="stage-toolbar">
            <div className="view-switcher">
              <button className={view === 'video' ? 'active' : ''} onClick={() => setView('video')}><VideoCamera /> Video</button>
              <button className={view === 'pitch' ? 'active' : ''} onClick={() => setView('pitch')}><SoccerBall /> Tactical pitch</button>
              <button className={view === 'analysis' ? 'active' : ''} onClick={() => setView('analysis')} disabled={!analysis}><ChartBar /> Analysis</button>
            </div>
            <div className="stage-actions">
              {videoAssetId && (
                <button className="detect-button" onClick={runDetector} disabled={mediaState === 'processing'}>
                  <Cpu /> {mediaState === 'processing' ? 'Processing…' : 'Run detector'}
                </button>
              )}
              {trackedVideoUrl && view === 'video' && <div className="source-switcher">
                <button className={videoSource === 'original' ? 'active' : ''} onClick={() => setVideoSource('original')}>Original</button>
                <button className={videoSource === 'tracked' ? 'active' : ''} onClick={() => setVideoSource('tracked')}>Tracked video</button>
              </div>}
              {analysis && view === 'video' && <button className={calibrationMode ? 'calibrate-button active' : 'calibrate-button'} onClick={() => { setCalibrationPoints([]); setCalibrationMode(true); setVideoSource('original') }}>
                <MapPin /> {calibrationMode ? `Mark point ${calibrationPoints.length + 1}/4` : activeCalibration ? `Add camera segment (${calibrations.length})` : 'Calibrate pitch'}
              </button>}
              {analysis && view === 'video' && <select className="calibration-scope" value={calibrationRegion} onChange={(event) => setCalibrationRegion(event.target.value as PitchRegion)} aria-label="Visible pitch region">
                {Object.entries(pitchRegions).map(([key, region]) => <option key={key} value={key}>{region.label} visible</option>)}
              </select>}
              <button onClick={() => fileRef.current?.click()}><FolderOpen /> Open video</button>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                onChange={(event) => onVideoFile(event.target.files?.[0])}
                hidden
              />
              <button aria-label="Full screen"><ArrowsOutSimple /></button>
            </div>
          </div>

          <div className="canvas-wrap">
            {view === 'analysis' && analysis ? (
              <section className="analysis-board">
                <div className="analysis-heading"><span>Computer vision report</span><strong>{analysis.summary.model}</strong></div>
                <div className="metric-grid">
                  <article><span>Stable track segments</span><strong>{analysis.summary.stable_track_segments?.player ?? analysis.summary.unique_tracks.player ?? 0}</strong><small>{analysis.summary.unique_tracks.player ?? 0} total fragments</small></article>
                  <article><span>Ball visibility</span><strong>{analysis.summary.ball_visible_percent}%</strong><small>{analysis.summary.detections.ball ?? 0} observations</small></article>
                  <article><span>Referee tracks</span><strong>{analysis.summary.unique_tracks.referee ?? 0}</strong><small>{analysis.summary.detections.referee ?? 0} observations</small></article>
                  <article><span>Frames analysed</span><strong>{analysis.summary.frames_sampled}</strong><small>{analysis.summary.image_size}px · stride {analysis.summary.sample_stride}</small></article>
                </div>
                <div className="activity-panel">
                  <div className="activity-title"><strong>Detection activity</strong><span>observations per minute</span></div>
                  <div className="activity-chart">
                    {analysis.activity.map((minute) => {
                      const peak = Math.max(...analysis.activity.map((item) => item.players), 1)
                      return <button key={minute.minute} onClick={() => seek(minute.minute * 60)} title={`Minute ${minute.minute + 1}: ${minute.players} player observations`}>
                        <i style={{ height: `${Math.max(5, minute.players / peak * 100)}%` }} />
                        <span>{minute.minute + 1}</span>
                      </button>
                    })}
                  </div>
                </div>
                <div className="analysis-foot"><span>{analysis.summary.device}</span><span>{analysis.summary.peak_vram_mb} MB peak VRAM</span><span>Passes: review candidates</span><span>Shots: awaiting pitch calibration</span></div>
              </section>
            ) : view === 'video' && videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoSource === 'tracked' && trackedVideoUrl ? trackedVideoUrl : videoUrl}
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onLoadedMetadata={(event) => registerVideo(event.currentTarget.duration || 600)}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                />
                {videoSource === 'original' && activeFrame && <div className="tracking-layer" aria-label="AI tracking overlay">
                  {activeFrame.d.map((detection, index) => <div
                    key={`${detection.id ?? index}-${detection.c}`}
                    className={`track-box ${detection.c} ${detection.team === 0 ? 'team-home' : detection.team === 1 ? 'team-away' : ''}`}
                    style={{ left: `${detection.b[0] * 100}%`, top: `${detection.b[1] * 100}%`,
                      width: `${(detection.b[2] - detection.b[0]) * 100}%`, height: `${(detection.b[3] - detection.b[1]) * 100}%` }}
                  ><span>{detection.c}{detection.id !== undefined ? ` ${detection.id}` : ''} · {Math.round(detection.p * 100)}%</span></div>)}
                </div>}
                {calibrationMode && <button className="calibration-layer" onClick={calibratePoint} aria-label={`Mark pitch corner ${calibrationPoints.length + 1}`}>
                  <span>{pitchRegions[calibrationRegion].label} at {formatTime(currentTime)} · mark its top-left, top-right, bottom-right, bottom-left intersections</span>
                  {calibrationPoints.map((point, index) => <i key={index} style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}>{index + 1}</i>)}
                </button>}
              </>
            ) : (
              <div className={view === 'pitch' ? 'pitch-canvas full-pitch' : 'pitch-canvas'}>
                <div className="pitch-lines">
                  <div className="halfway" />
                  <div className="center-circle" />
                  <div className="box left" />
                  <div className="box right" />
                </div>
                {homography && <svg className="pitch-trails" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {Array.from(pitchTrails.entries()).map(([id, points]) => <polyline key={id} points={points.map((point) => `${point.x},${point.y}`).join(' ')} />)}
                </svg>}
                {livePositions.map(({ id, x, y, team }) => (
                  <button
                    key={id}
                    className={`player-marker ${team === 0 ? 'home' : 'away'}`}
                    style={{ left: `${7 + x * .86}%`, top: `${7 + y * .86}%` }}
                    aria-label={`Track ${id}`}
                  >
                    {id}
                  </button>
                ))}
                {liveBall && <span className="pitch-ball" style={{ left: `${7 + liveBall.x * .86}%`, top: `${7 + liveBall.y * .86}%` }} />}
                {view === 'pitch' && !homography && <div className="pitch-empty"><MapPin /><strong>No camera calibration at {formatTime(currentTime)}</strong><span>Open Video, choose the visible pitch region and add a calibration segment at this camera view.</span></div>}
                {view === 'pitch' && activeCalibration && <div className="pitch-calibration-status">{pitchRegions[activeCalibration.region].label} · calibrated at {formatTime(activeCalibration.time)} · segment {calibrations.indexOf(activeCalibration) + 1}/{calibrations.length}</div>}
                {view === 'pitch' && activeCalibration && <div className="pitch-coverage">
                  <span>{teamCode(homeTeam)} {visibleTeamCounts.home} visible</span>
                  <span>{teamCode(awayTeam)} {visibleTeamCounts.away} visible</span>
                  <strong>{activeCalibration.region === 'full' && visibleTeamCounts.home >= 8 && visibleTeamCounts.away >= 8 ? 'Shape sample ready' : 'Partial view · formation withheld'}</strong>
                </div>}
                {!videoUrl && view === 'video' && (
                  <button className="video-empty" onClick={() => fileRef.current?.click()}>
                    <span className="upload-icon"><UploadSimple /></span>
                    <strong>Open a match recording</strong>
                    <span>MP4, MOV or WebM. Start with a 10-minute segment.</span>
                  </button>
                )}
              </div>
            )}

            <div className="scorebug">
              <span>{formatTime(currentTime)}</span>
              <strong>{teamCode(homeTeam)}</strong><b>0</b><b>0</b><strong>{teamCode(awayTeam)}</strong>
            </div>
            <div className="canvas-caption">
              <span>{fileName}</span>
              <span>{mediaState === 'uploading' ? 'Uploading to local storage…' : mediaState === 'ready' ? 'Ready for detection' : mediaState === 'processing' ? 'Detection running…' : mediaState === 'done' ? processingSummary : mediaState === 'error' ? processingSummary : `Source time ${formatTime(currentTime)}`}</span>
            </div>
          </div>

          <div className="transport">
            <button className="play-button" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
              {playing ? <Pause weight="fill" /> : <Play weight="fill" />}
            </button>
            <button className="skip-button" onClick={() => seek(currentTime + 5)} aria-label="Skip forward 5 seconds"><FastForward /></button>
            <time>{formatTime(currentTime)} <span>/ {formatTime(duration)}</span></time>
            <input
              className="scrubber"
              type="range"
              min="0"
              max={duration}
              step="0.04"
              value={Math.min(currentTime, duration)}
              onChange={(event) => seek(Number(event.target.value))}
              style={{ '--progress': `${(currentTime / duration) * 100}%` } as React.CSSProperties}
              aria-label="Video timeline"
            />
            <button className="speed-button">1x <CaretDown /></button>
          </div>

          <div className="event-track">
            <div className="track-label">EVENTS</div>
            <div className="track-line">
              {events.map((event) => (
                <button
                  key={event.id}
                  className={`track-event ${event.kind.toLowerCase()} ${selectedEvent === event.id ? 'selected' : ''}`}
                  style={{ left: `${Math.min((event.time / duration) * 100, 100)}%` }}
                  onClick={() => { setSelectedEvent(event.id); seek(event.time) }}
                  aria-label={`${event.kind} at ${formatTime(event.time)}`}
                >
                  <span />
                </button>
              ))}
              <div className="playhead" style={{ left: `${Math.min((currentTime / duration) * 100, 100)}%` }} />
            </div>
          </div>

          <div className="quick-events">
            <span>Code event</span>
            {(['Pass', 'Shot', 'Recovery', 'Press', 'Note'] as EventKind[]).map((kind, index) => (
              <button key={kind} onClick={() => addEvent(kind)}>
                <kbd>{index + 1}</kbd>{kind}
              </button>
            ))}
          </div>
        </section>

        <aside className="inspector">
          <div className="inspector-head">
            <div>
              <h2>Events</h2>
            </div>
            <button className="add-button" onClick={() => addEvent('Note')}><Plus /> Add</button>
          </div>

          <label className="search-box">
            <MagnifyingGlass />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search events or players" />
            {query && <button onClick={() => setQuery('')} aria-label="Clear search"><X /></button>}
          </label>

          <div className="filter-row">
            <button className="active">All <span>{events.length}</span></button>
            <button>Review <span>{events.filter((event) => event.status === 'review').length}</span></button>
            <button>Confirmed</button>
          </div>

          <div className="event-list">
            {filteredEvents.length ? filteredEvents.map((event) => (
              <button
                key={event.id}
                className={selectedEvent === event.id ? 'event-row selected' : 'event-row'}
                onClick={() => { setSelectedEvent(event.id); seek(event.time) }}
              >
                <span className={`event-glyph ${event.kind.toLowerCase()}`}>
                  {event.kind === 'Shot' ? <SoccerBall weight="fill" /> : event.kind === 'Press' ? <UsersThree weight="fill" /> : <Flag weight="fill" />}
                </span>
                <span className="event-copy">
                  <strong>{event.kind}<small>{event.status === 'review' ? 'Needs review' : 'Confirmed'}</small></strong>
                  <span>{event.player} / {event.team}</span>
                </span>
                <time>{formatTime(event.time)}</time>
              </button>
            )) : (
              <div className="empty-results"><MagnifyingGlass /><strong>No matching events</strong><span>Try a player, team or event type.</span></div>
            )}
          </div>

          {activeEvent && (
            <div className="event-detail">
              <div className="detail-heading">
                <div><h3>{activeEvent.kind} at {formatTime(activeEvent.time)}</h3><span className="detail-source">Analyst event</span></div>
                <ArrowUpRight />
              </div>
              <p>{activeEvent.note}</p>
              <dl>
                <div><dt>Player</dt><dd>{activeEvent.player}</dd></div>
                <div><dt>Source</dt><dd>Analyst</dd></div>
              </dl>
              <div className="detail-actions">
                <button className="confirm-button" onClick={confirmSelected}><Check weight="bold" /> Confirm</button>
                <button className="delete-button" onClick={deleteSelected} aria-label="Delete event"><Trash /></button>
              </div>
            </div>
          )}
        </aside>
      </section>

      <footer>
        <span>{connection === 'connected' ? 'API workspace' : 'Local workspace'}</span>
        <span>{connection === 'connected' ? 'Revision history enabled' : 'Browser storage'}</span>
      </footer>
    </main>
  )
}

export default App





