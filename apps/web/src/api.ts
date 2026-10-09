export type ApiEvent = {
  id: string
  timestamp_ms: number
  event_type: string
  team_name: string | null
  player_name: string | null
  source: string
  verification_status: string
  note: string | null
}

type Organization = { id: string; name: string; slug: string }
export type Match = { id: string; organization_id: string; home_team: string; away_team: string }
export type ApiVideo = { id: string; filename: string; duration_ms: number; storage_key: string }
export type ProcessingJob = {
  id: string; status: 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED'; progress: number
  mode: 'MOCK' | 'YOLO'; result_summary: string | null; error_message: string | null
}
export type TrackDetection = { c: 'player' | 'ball' | 'referee'; p: number; b: [number, number, number, number]; id?: number; team?: 0 | 1 }
export type AnalysisData = {
  version: number
  video: { width: number; height: number; fps: number; duration_seconds: number }
  summary: {
    frames_sampled: number; sample_stride: number; source_fps: number; duration_seconds: number
    detections: Record<string, number>; unique_tracks: Record<string, number>; stable_track_segments?: Record<string, number>
    average_confidence: Record<string, number>; ball_visible_percent: number
    most_active_track: number | null; peak_vram_mb: number; device: string; model: string; image_size: number
    team_tracks?: { home: number; away: number }; event_candidates?: Record<string, number>; tracking_scope?: string; tracker?: string; shot_detection?: string; camera_cuts?: number[]
  }
  activity: { minute: number; players: number; balls: number; referees: number }[]
  events: { type: 'PASS_CANDIDATE' | 'SHOT_CANDIDATE'; t: number; team?: number; confidence: number; status: 'REVIEW'; from_track?: number; to_track?: number; track?: number }[]
  frames: { t: number; d: TrackDetection[] }[]
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json() as Promise<T>
}

async function findOrCreateOrganization(): Promise<Organization> {
  const response = await fetch(`${API_URL}/api/v1/organizations/by-slug/touchline-development`)
  if (response.ok) return response.json() as Promise<Organization>
  if (response.status !== 404) throw new Error(`${response.status}: ${await response.text()}`)
  return request<Organization>('/api/v1/organizations', {
    method: 'POST',
    body: JSON.stringify({ name: 'Touchline Development', slug: 'touchline-development' }),
  })
}

export async function bootstrapWorkspace() {
  const organization = await findOrCreateOrganization()
  const headers = { 'X-Organization-ID': organization.id }
  const matches = await request<Match[]>('/api/v1/matches', { headers })
  const match = matches[0] ?? await request<Match>('/api/v1/matches', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      home_team: 'KMC FC',
      away_team: 'Coastal Union',
      competition: 'Development fixture',
    }),
  })
  const events = await request<ApiEvent[]>(`/api/v1/matches/${match.id}/events`, { headers })
  return { organization, match, events }
}

export function createApiEvent(organizationId: string, matchId: string, event: Omit<ApiEvent, 'id'>) {
  return request<ApiEvent>(`/api/v1/matches/${matchId}/events`, {
    method: 'POST',
    headers: { 'X-Organization-ID': organizationId },
    body: JSON.stringify(event),
  })
}

export function updateApiEvent(organizationId: string, eventId: string, changes: Record<string, unknown>) {
  return request<ApiEvent>(`/api/v1/events/${eventId}`, {
    method: 'PATCH',
    headers: { 'X-Organization-ID': organizationId },
    body: JSON.stringify({ ...changes, changed_by: 'local-analyst' }),
  })
}

export async function uploadApiVideo(
  organizationId: string, matchId: string, file: File, durationMs: number,
) {
  const body = new FormData()
  body.append('file', file)
  body.append('duration_ms', String(durationMs))
  body.append('source_start_ms', '0')
  const response = await fetch(`${API_URL}/api/v1/matches/${matchId}/media`, {
    method: 'POST', headers: { 'X-Organization-ID': organizationId }, body,
  })
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json() as Promise<ApiVideo>
}

export function createProcessingJob(organizationId: string, matchId: string, videoId: string, mode: 'MOCK' | 'YOLO' = 'YOLO') {
  return request<ProcessingJob>(`/api/v1/matches/${matchId}/jobs`, {
    method: 'POST', headers: { 'X-Organization-ID': organizationId },
    body: JSON.stringify({ video_id: videoId, mode }),
  })
}

export function getProcessingJob(organizationId: string, jobId: string) {
  return request<ProcessingJob>(`/api/v1/jobs/${jobId}`, {
    headers: { 'X-Organization-ID': organizationId },
  })
}

export function getJobAnalysis(organizationId: string, jobId: string) {
  return request<AnalysisData>(`/api/v1/jobs/${jobId}/analysis`, {
    headers: { 'X-Organization-ID': organizationId },
  })
}

export function updateMatch(
  organizationId: string, matchId: string, changes: { home_team?: string; away_team?: string },
) {
  return request<Match>(`/api/v1/matches/${matchId}`, {
    method: 'PATCH', headers: { 'X-Organization-ID': organizationId }, body: JSON.stringify(changes),
  })
}

export async function getTrackedVideo(organizationId: string, jobId: string) {
  const response = await fetch(`${API_URL}/api/v1/jobs/${jobId}/tracked-video`, {
    headers: { 'X-Organization-ID': organizationId },
  })
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.blob()
}
